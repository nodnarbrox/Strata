// Native, single-sequence CPU LFM2-MoE graph for Strata. No llama_decode/server.
// Architecture reference: llama.cpp 8642541b (MIT), src/models/lfm*.cpp.
// The graph uses the selected GGML CPU kernels and Strata's mmap reader. State belongs
// to this engine: gated B*X history and F16 attention K/V are reset together.
#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define SL_API extern "C" __declspec(dllexport)
#else
#define SL_API extern "C" __attribute__((visibility("default")))
#endif

namespace {
thread_local std::string error;
void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}
struct Context {
    ggml_context * p = nullptr;
    Context(size_t bytes, bool no_alloc) {
        p = ggml_init({bytes, nullptr, no_alloc});
        require(p != nullptr, "GGML context allocation failed");
    }
    ~Context() { if (p) ggml_free(p); }
};
struct LayerState {
    std::vector<float> conv;
    std::vector<ggml_fp16_t> k, v;
    ggml_tensor * history = nullptr, * keys = nullptr, * values = nullptr;
};
struct Engine {
    strata::GgufFile file;
    Context weights{8 * 1024 * 1024, true};
    Context states{2 * 1024 * 1024, true};
    Context graph{64 * 1024 * 1024, false};
    std::map<std::string, ggml_tensor *> tensors;
    std::vector<LayerState> layers;
    std::vector<int> heads, kvheads;
    std::vector<uint8_t> workspace;
    std::vector<std::string> vocabulary;
    std::vector<float> last_layers;
    ggml_threadpool * pool = nullptr;
    int d, nl, vocab, dense, experts, used, cache, ctx, threads, position = 0;
    int trained_ctx, rot, gating;
    float eps, freq, expert_scale;

    double number(const std::string & key, double fallback = -1) const {
        const auto * m = file.get("lfm2moe." + key);
        if (!m) { require(fallback >= 0, "Missing Liquid metadata: " + key); return fallback; }
        require(m->is_num(), "Expected scalar Liquid metadata: " + key);
        return m->num();
    }
    std::vector<int> integers(const std::string & key) const {
        const auto * m = file.get("lfm2moe." + key);
        require(m, "Missing Liquid metadata: " + key);
        if (m->type != strata::MetaType::ARRAY) return std::vector<int>(nl, int(m->num()));
        require(m->items.size() == size_t(nl), "Layer metadata length mismatch: " + key);
        std::vector<int> out;
        for (const auto & item : m->items) out.push_back(int(item.num()));
        return out;
    }
    ggml_tensor * weight(const std::string & name) const {
        const auto it = tensors.find(name);
        require(it != tensors.end(), "Missing Liquid tensor: " + name);
        return it->second;
    }
    ggml_tensor * w(int il, const char * suffix) const {
        return weight("blk." + std::to_string(il) + "." + suffix);
    }
    void shape(ggml_tensor * t, std::initializer_list<int64_t> expected) const {
        int i = 0;
        for (int64_t n : expected) require(t->ne[i++] == n, std::string("Wrong tensor shape: ") + t->name);
        for (; i < 4; ++i) require(t->ne[i] == 1, std::string("Wrong tensor rank: ") + t->name);
    }
    Engine(const std::string & path, int context, int nthreads) : file(path), ctx(context), threads(nthreads) {
        const auto * arch = file.get("general.architecture");
        require(arch && arch->s == "lfm2moe", "strata-liquid requires architecture lfm2moe");
        require(!file.get("split.count") || file.get("split.count")->num() <= 1, "Split GGUF is not supported by this CPU port");
        d = int(number("embedding_length")); nl = int(number("block_count"));
        dense = int(number("leading_dense_block_count", 0));
        experts = int(number("expert_count")); used = int(number("expert_used_count"));
        cache = int(number("shortconv.l_cache"));
        eps = float(number("attention.layer_norm_rms_epsilon"));
        freq = float(number("rope.freq_base", 10000));
        trained_ctx = int(number("context_length"));
        rot = int(number("rope.dimension_count", 0));
        gating = int(number("expert_gating_func"));
        expert_scale = float(number("expert_weights_scale", 1));
        require(d > 0 && d <= 65536 && nl > 0 && nl <= 256 && dense >= 0 && dense <= nl, "Invalid Liquid geometry");
        require(cache >= 2 && cache <= 64 && used > 0 && used <= experts && experts <= 1024, "Invalid convolution/expert geometry");
        require(ctx > 0 && ctx <= trained_ctx && threads > 0 && threads <= 256, "Context/threads outside supported bounds");
        require(gating == 1 || gating == 2, "Only softmax or sigmoid expert routing is supported");
        require(number("attention.sliding_window", 0) == 0, "Sliding-window variants are not supported");
        require(number("rope.scaling.factor", 1) == 1, "Scaled RoPE variants are not supported");
        require(number("attention.causal", 1) == 1, "Non-causal variants are not supported");
        heads = integers("attention.head_count"); kvheads = integers("attention.head_count_kv");
        for (const auto & t : file.tensors()) {
            require(t.shape.size() >= 1 && t.shape.size() <= 4, "Invalid tensor rank: " + t.name);
            const uint64_t bytes = strata::tensor_payload_bytes(t);
            require(bytes && t.offset <= file.file_size() - file.data_start() &&
                    bytes <= file.file_size() - file.data_start() - t.offset, "Truncated/unsupported tensor: " + t.name);
            require(t.type < GGML_TYPE_COUNT && ggml_blck_size(ggml_type(t.type)) > 0, "Unsupported GGML type: " + t.name);
            int64_t sizes[4] = {1, 1, 1, 1};
            for (size_t i = 0; i < t.shape.size(); ++i) sizes[i] = int64_t(t.shape[i]);
            auto * descriptor = ggml_new_tensor(weights.p, ggml_type(t.type), int(t.shape.size()), sizes);
            descriptor->data = const_cast<uint8_t *>(file.tensor_data(t));
            ggml_set_name(descriptor, t.name.c_str());
            require(tensors.emplace(t.name, descriptor).second, "Duplicate tensor: " + t.name);
        }
        auto * emb = weight("token_embd.weight"); vocab = int(emb->ne[1]); shape(emb, {d, vocab});
        // The shared inspection reader retains only 64 array entries. Read the
        // complete tokenizer here without changing other model families' reader.
        require(file.data_start() <= 256 * 1024 * 1024, "Liquid GGUF header is too large");
        std::vector<uint8_t> header(size_t(file.data_start()));
        std::ifstream stream(path, std::ios::binary);
        stream.read(reinterpret_cast<char *>(header.data()), std::streamsize(header.size()));
        require(bool(stream), "Cannot read complete Liquid GGUF header");
        strata::Cursor cursor(header.data(), header.size());
        cursor.read<uint32_t>(); cursor.read<uint32_t>(); cursor.read<uint64_t>();
        const auto nkv = cursor.read<uint64_t>();
        for (uint64_t i = 0; i < nkv; ++i) {
            const auto key = cursor.str(); const auto type = strata::MetaType(cursor.read<uint32_t>());
            if (key == "tokenizer.ggml.tokens") {
                require(type == strata::MetaType::ARRAY && cursor.read<uint32_t>() == uint32_t(strata::MetaType::STRING), "Invalid tokenizer vocabulary");
                const auto count = cursor.read<uint64_t>(); require(count == uint64_t(vocab), "Tokenizer/model vocabulary sizes differ");
                vocabulary.reserve(vocab);
                for (int j = 0; j < vocab; ++j) vocabulary.push_back(cursor.str());
            } else { strata::read_value(cursor, type); }
        }
        shape(weight("token_embd_norm.weight"), {d});
        if (!tensors.count("output.weight")) tensors["output.weight"] = emb;
        shape(weight("output.weight"), {d, vocab});
        layers.resize(nl);
        const auto ffn = integers("feed_forward_length");
        const auto effn = integers("expert_feed_forward_length");
        for (int il = 0; il < nl; ++il) {
            shape(w(il, "attn_norm.weight"), {d}); shape(w(il, "ffn_norm.weight"), {d});
            auto & s = layers[il];
            if (kvheads[il]) {
                require(heads[il] > 0 && d % heads[il] == 0 && heads[il] % kvheads[il] == 0, "Invalid GQA head counts");
                const int hd = d / heads[il], kd = hd * kvheads[il];
                require(hd % 2 == 0 && (!rot || rot == hd), "Partial/odd rotary dimensions are not supported");
                shape(w(il, "attn_q.weight"), {d, d}); shape(w(il, "attn_k.weight"), {d, kd});
                shape(w(il, "attn_v.weight"), {d, kd}); shape(w(il, "attn_output.weight"), {d, d});
                shape(w(il, "attn_q_norm.weight"), {hd}); shape(w(il, "attn_k_norm.weight"), {hd});
                s.k.resize(size_t(kd) * ctx); s.v.resize(size_t(kd) * ctx);
                s.keys = ggml_new_tensor_3d(states.p, GGML_TYPE_F16, hd, ctx, kvheads[il]); s.keys->data = s.k.data();
                s.values = ggml_new_tensor_3d(states.p, GGML_TYPE_F16, hd, ctx, kvheads[il]); s.values->data = s.v.data();
            } else {
                shape(w(il, "shortconv.in_proj.weight"), {d, 3 * d});
                shape(w(il, "shortconv.out_proj.weight"), {d, d}); shape(w(il, "shortconv.conv.weight"), {cache, d});
                require(w(il, "shortconv.conv.weight")->type == GGML_TYPE_F32, "Convolution kernel must be F32");
                s.conv.resize(size_t(cache - 1) * d);
                s.history = ggml_new_tensor_3d(states.p, GGML_TYPE_F32, cache - 1, d, 1); s.history->data = s.conv.data();
            }
            if (il < dense) {
                shape(w(il, "ffn_up.weight"), {d, ffn[il]}); shape(w(il, "ffn_gate.weight"), {d, ffn[il]});
                shape(w(il, "ffn_down.weight"), {ffn[il], d});
            } else {
                shape(w(il, "ffn_gate_inp.weight"), {d, experts}); shape(w(il, "exp_probs_b.bias"), {experts});
                shape(w(il, "ffn_up_exps.weight"), {d, effn[il], experts});
                shape(w(il, "ffn_gate_exps.weight"), {d, effn[il], experts});
                shape(w(il, "ffn_down_exps.weight"), {effn[il], d, experts});
            }
        }
        auto params = ggml_threadpool_params_default(threads);
        pool = ggml_threadpool_new(&params);
        require(pool, "CPU threadpool allocation failed");
    }
    ~Engine() { if (pool) ggml_threadpool_free(pool); }
    void reset() {
        last_layers.clear();
        for (auto & s : layers) {
            std::fill(s.conv.begin(), s.conv.end(), 0); std::fill(s.k.begin(), s.k.end(), 0); std::fill(s.v.begin(), s.v.end(), 0);
        }
        position = 0;
    }
    ggml_tensor * norm(ggml_tensor * x, ggml_tensor * scale) {
        return ggml_mul(graph.p, ggml_rms_norm(graph.p, x, eps), scale);
    }
    ggml_tensor * mm(ggml_tensor * a, ggml_tensor * b) { return ggml_mul_mat(graph.p, a, b); }
    void decode(int32_t token, float * output) {
        require(position < ctx, "Liquid context is full; reset before another request");
        require(token >= 0 && token < vocab, "Token ID outside Liquid vocabulary");
        ggml_reset(graph.p);
        auto * g = graph.p;
        auto * gf = ggml_new_graph_custom(g, 8192, false);
        auto * tid = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1); std::memcpy(tid->data, &token, sizeof token);
        auto * pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1); std::memcpy(pos->data, &position, sizeof position);
        auto * cur = ggml_get_rows(g, weight("token_embd.weight"), tid);
        std::vector<ggml_tensor *> layer_outputs;
        for (int il = 0; il < nl; ++il) {
            auto * residual = cur;
            cur = norm(cur, w(il, "attn_norm.weight"));
            auto & s = layers[il];
            if (kvheads[il]) {
                const int hd = d / heads[il], hk = kvheads[il];
                auto * q = ggml_reshape_3d(g, mm(w(il, "attn_q.weight"), cur), hd, heads[il], 1);
                auto * k = ggml_reshape_3d(g, mm(w(il, "attn_k.weight"), cur), hd, hk, 1);
                auto * v = ggml_reshape_3d(g, mm(w(il, "attn_v.weight"), cur), hd, hk, 1);
                q = norm(q, w(il, "attn_q_norm.weight")); k = norm(k, w(il, "attn_k_norm.weight"));
                q = ggml_rope_ext(g, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, trained_ctx, freq, 1, 0, 1, 32, 1);
                k = ggml_rope_ext(g, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, trained_ctx, freq, 1, 0, 1, 32, 1);
                q = ggml_permute(g, q, 0, 2, 1, 3); k = ggml_permute(g, k, 0, 2, 1, 3); v = ggml_permute(g, v, 0, 2, 1, 3);
                const size_t offset = size_t(position) * hd * sizeof(ggml_fp16_t);
                auto * kwrite = ggml_view_3d(g, s.keys, hd, 1, hk, s.keys->nb[1], s.keys->nb[2], offset);
                auto * vwrite = ggml_view_3d(g, s.values, hd, 1, hk, s.values->nb[1], s.values->nb[2], offset);
                ggml_build_forward_expand(gf, ggml_cpy(g, k, kwrite)); ggml_build_forward_expand(gf, ggml_cpy(g, v, vwrite));
#ifdef STRATA_LIQUID_REFERENCE_KV_PADDING
                // Match the reference CPU attention reduction geometry at cache boundaries.
                const int n_kv = std::min(ctx, ((position + 1 + 255) / 256) * 256);
                auto * kr = ggml_view_3d(g, s.keys, hd, n_kv, hk, s.keys->nb[1], s.keys->nb[2], 0);
                auto * vr = ggml_view_3d(g, s.values, hd, n_kv, hk, s.values->nb[1], s.values->nb[2], 0);
                auto * mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, n_kv, 1);
                auto * mask_data = static_cast<ggml_fp16_t *>(mask->data);
                for (int i = 0; i < n_kv; ++i) mask_data[i] = ggml_fp32_to_fp16(i <= position ? 0.0f : -INFINITY);
                auto * attn = ggml_flash_attn_ext(g, q, kr, vr, mask, 1.0f / std::sqrt(float(hd)), 0, 0);
#else
                auto * kr = ggml_view_3d(g, s.keys, hd, position + 1, hk, s.keys->nb[1], s.keys->nb[2], 0);
                auto * vr = ggml_view_3d(g, s.values, hd, position + 1, hk, s.values->nb[1], s.values->nb[2], 0);
                auto * attn = ggml_flash_attn_ext(g, q, kr, vr, nullptr, 1.0f / std::sqrt(float(hd)), 0, 0);
#endif
                ggml_prec_set_acc(attn, GGML_PREC_F32);
                cur = mm(w(il, "attn_output.weight"), ggml_reshape_2d(g, attn, d, 1));
            } else {
                auto * bcx = mm(w(il, "shortconv.in_proj.weight"), cur);
                auto * b = ggml_view_3d(g, bcx, d, 1, 1, bcx->nb[1], bcx->nb[2], 0);
                auto * c = ggml_view_3d(g, bcx, d, 1, 1, bcx->nb[1], bcx->nb[2], size_t(d) * sizeof(float));
                auto * x = ggml_view_3d(g, bcx, d, 1, 1, bcx->nb[1], bcx->nb[2], size_t(2 * d) * sizeof(float));
                auto * bx = ggml_concat(g, s.history, ggml_transpose(g, ggml_mul(g, b, x)), 0);
                auto * next = ggml_view_3d(g, bx, cache - 1, d, 1, bx->nb[1], bx->nb[2], sizeof(float));
                ggml_build_forward_expand(gf, ggml_cpy(g, next, s.history));
                auto * conv = ggml_ssm_conv(g, bx, w(il, "shortconv.conv.weight"));
                cur = ggml_reshape_2d(g, mm(w(il, "shortconv.out_proj.weight"), ggml_mul(g, c, conv)), d, 1);
            }
            cur = ggml_add(g, residual, cur); residual = cur;
            auto * n = norm(cur, w(il, "ffn_norm.weight"));
            if (il < dense) {
                cur = mm(w(il, "ffn_down.weight"), ggml_swiglu_split(g, mm(w(il, "ffn_gate.weight"), n), mm(w(il, "ffn_up.weight"), n)));
            } else {
                auto * logits = mm(w(il, "ffn_gate_inp.weight"), n);
                auto * probs = gating == 2 ? ggml_sigmoid(g, logits) : ggml_soft_max(g, logits);
                auto * ids = ggml_argsort_top_k(g, ggml_add(g, probs, w(il, "exp_probs_b.bias")), used);
                auto * route = ggml_get_rows(g, ggml_reshape_3d(g, probs, 1, experts, 1), ids);
                route = ggml_reshape_2d(g, route, used, 1);
                route = ggml_div(g, route, ggml_clamp(g, ggml_sum_rows(g, route), 6.103515625e-5f, INFINITY));
                if (expert_scale != 0 && expert_scale != 1) route = ggml_scale(g, route, expert_scale);
                route = ggml_reshape_3d(g, route, 1, used, 1); ggml_build_forward_expand(gf, route);
                auto * input = ggml_reshape_3d(g, n, d, 1, 1);
                auto * up = ggml_mul_mat_id(g, w(il, "ffn_up_exps.weight"), input, ids);
                auto * gate = ggml_mul_mat_id(g, w(il, "ffn_gate_exps.weight"), input, ids);
                auto * ff = ggml_swiglu_split(g, gate, up);
                auto * down = ggml_mul(g, ggml_mul_mat_id(g, w(il, "ffn_down_exps.weight"), ff, ids), route);
                // Match the reference's ordered accumulation across chosen experts.
                cur = ggml_view_2d(g, down, d, 1, down->nb[2], 0);
                for (int j = 1; j < used; ++j) cur = ggml_add(g, cur, ggml_view_2d(g, down, d, 1, down->nb[2], j * down->nb[1]));
            }
            cur = ggml_add(g, residual, cur);
            layer_outputs.push_back(cur);
        }
        cur = mm(weight("output.weight"), norm(cur, weight("token_embd_norm.weight")));
        ggml_build_forward_expand(gf, cur);
        auto plan = ggml_graph_plan(gf, threads, pool);
        workspace.resize(plan.work_size); plan.work_data = workspace.data();
        require(ggml_graph_compute(gf, &plan) == GGML_STATUS_SUCCESS, "Liquid graph execution failed");
        std::memcpy(output, cur->data, size_t(vocab) * sizeof(float));
        last_layers.resize(size_t(nl) * d);
        for (int il = 0; il < nl; ++il) std::memcpy(last_layers.data() + size_t(il) * d, layer_outputs[il]->data, size_t(d) * sizeof(float));
        require(std::all_of(output, output + vocab, [](float x) { return std::isfinite(x); }), "Non-finite Liquid logits");
        ++position;
    }
};
}

SL_API const char * sl_error() { return error.c_str(); }
SL_API void * sl_create(const char * path, int context, int threads) {
    try { error.clear(); require(path, "Model path is required"); return new Engine(path, context, threads); }
    catch (const std::exception & e) { error = e.what(); return nullptr; }
}
SL_API void sl_free(void * handle) { delete static_cast<Engine *>(handle); }
SL_API int sl_vocab(void * handle) { return handle ? static_cast<Engine *>(handle)->vocab : 0; }
SL_API int sl_position(void * handle) { return handle ? static_cast<Engine *>(handle)->position : -1; }
SL_API int sl_layer(void * handle, int layer, float * values, int size) {
    auto * e = static_cast<Engine *>(handle);
    if (!e || !values || size != e->d || layer < 0 || layer >= e->nl || e->last_layers.empty()) return -1;
    std::memcpy(values, e->last_layers.data() + size_t(layer) * e->d, size_t(e->d) * sizeof(float)); return 0;
}
SL_API const char * sl_meta_string(void * handle, const char * key) {
    if (!handle || !key) return nullptr;
    const auto * value = static_cast<Engine *>(handle)->file.get(key);
    return value && value->type == strata::MetaType::STRING ? value->s.c_str() : nullptr;
}
SL_API int sl_meta_int(void * handle, const char * key) {
    if (!handle || !key) return -1;
    const auto * value = static_cast<Engine *>(handle)->file.get(key);
    return value && value->is_num() ? int(value->num()) : -1;
}
SL_API const char * sl_vocab_text(void * handle, int token) {
    if (!handle || token < 0) return nullptr;
    const auto & value = static_cast<Engine *>(handle)->vocabulary;
    return size_t(token) < value.size() ? value[token].c_str() : nullptr;
}
SL_API int sl_reset(void * handle) {
    if (!handle) { error = "Missing Liquid handle"; return -1; }
    static_cast<Engine *>(handle)->reset(); return 0;
}
SL_API int sl_decode(void * handle, const int32_t * tokens, int count, float * logits, int size) {
    try {
        auto * e = static_cast<Engine *>(handle);
        require(e && tokens && logits && count > 0, "Invalid Liquid decode arguments");
        require(size == e->vocab, "Logit buffer must match vocabulary");
        require(count <= e->ctx - e->position, "Liquid context capacity exceeded");
        for (int i = 0; i < count; ++i) require(tokens[i] >= 0 && tokens[i] < e->vocab, "Token ID outside vocabulary");
        for (int i = 0; i < count; ++i) e->decode(tokens[i], logits);
        return 0;
    } catch (const std::exception & ex) { error = ex.what(); return -1; }
}
