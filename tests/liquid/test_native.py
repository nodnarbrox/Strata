"""Independent NumPy oracle: convolution history, GQA/RoPE, routing and resets.

Runs a complete mixed architecture, rather than tests that restate C++ helpers.
Requires a built strata-liquid library. Synthetic GGUF contains no real weights.
"""
import struct
import numpy as np
import pytest

from strata_control.liquid_native import LiquidEngine


def string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def make_model(path, *, architecture="lfm2moe", gating=2):
    rng = np.random.default_rng(3917)
    d, nl, h, hk, ff, ef, nex, top, nv = 64, 4, 2, 1, 96, 64, 4, 2, 48
    meta = {"general.architecture": architecture, "lfm2moe.embedding_length": d,
            "lfm2moe.block_count": nl, "lfm2moe.leading_dense_block_count": 1,
            "lfm2moe.expert_count": nex, "lfm2moe.expert_used_count": top,
            "lfm2moe.expert_gating_func": gating, "lfm2moe.shortconv.l_cache": 3,
            "lfm2moe.attention.layer_norm_rms_epsilon": 1e-5, "lfm2moe.rope.freq_base": 5000000.,
            "lfm2moe.context_length": 64, "lfm2moe.attention.head_count": [h] * nl,
            "lfm2moe.attention.head_count_kv": [0, hk, 0, hk],
            "lfm2moe.feed_forward_length": ff, "lfm2moe.expert_feed_forward_length": ef,
            "lfm2moe.expert_weights_scale": 1.3}
    weights = {}
    def add(name, shape, norm=False):
        weights[name] = (np.ones(shape) if norm else rng.normal(0, .08, shape)).astype(np.float32)
    add("token_embd.weight", (nv, d)); add("token_embd_norm.weight", (d,), True)
    add("output.weight", (nv, d))
    for il in range(nl):
        prefix = f"blk.{il}."
        add(prefix + "attn_norm.weight", (d,), True); add(prefix + "ffn_norm.weight", (d,), True)
        if il % 2:
            for suffix, shape in [("attn_q", (d, d)), ("attn_k", (d // h * hk, d)),
                                  ("attn_v", (d // h * hk, d)), ("attn_output", (d, d))]:
                add(prefix + suffix + ".weight", shape)
            add(prefix + "attn_q_norm.weight", (d // h,), True); add(prefix + "attn_k_norm.weight", (d // h,), True)
        else:
            add(prefix + "shortconv.in_proj.weight", (3 * d, d))
            add(prefix + "shortconv.out_proj.weight", (d, d))
            add(prefix + "shortconv.conv.weight", (d, 3))
        if il == 0:
            add(prefix + "ffn_up.weight", (ff, d)); add(prefix + "ffn_gate.weight", (ff, d))
            add(prefix + "ffn_down.weight", (d, ff))
        else:
            add(prefix + "ffn_gate_inp.weight", (nex, d)); add(prefix + "exp_probs_b.bias", (nex,))
            add(prefix + "ffn_up_exps.weight", (nex, ef, d)); add(prefix + "ffn_gate_exps.weight", (nex, ef, d))
            add(prefix + "ffn_down_exps.weight", (nex, d, ef))
    header = b"GGUF" + struct.pack("<IQQ", 3, len(weights), len(meta))
    for key, v in meta.items():
        header += string(key)
        if isinstance(v, str): header += struct.pack("<I", 8) + string(v)
        elif isinstance(v, list): header += struct.pack("<IIQ", 9, 4, len(v)) + struct.pack("<" + "I" * len(v), *v)
        elif isinstance(v, float): header += struct.pack("<If", 6, v)
        else: header += struct.pack("<II", 4, v)
    data = b""
    for name, a in weights.items():
        header += string(name) + struct.pack("<I", a.ndim) + struct.pack("<" + "Q" * a.ndim, *a.shape[::-1]) + struct.pack("<IQ", 0, len(data))
        data += a.tobytes(); data += b"\0" * (-len(data) % 32)
    path.write_bytes(header + b"\0" * (-len(header) % 32) + data)
    return weights


def reference(weights, tokens, gating=2):
    history = [np.zeros((64, 2), np.float32) for _ in range(4)]
    keys, values = {}, {}
    def rms(a, w): return a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + 1e-5) * w
    def silu(a): return a / (1 + np.exp(-a))
    all_logits = []
    for pos, token in enumerate(tokens):
        cur = weights["token_embd.weight"][token].copy()
        for il in range(4):
            w = lambda suffix: weights[f"blk.{il}." + suffix]
            n = rms(cur, w("attn_norm.weight"))
            if il % 2:
                q = rms((w("attn_q.weight") @ n).reshape(2, 32), w("attn_q_norm.weight"))
                k = rms((w("attn_k.weight") @ n).reshape(1, 32), w("attn_k_norm.weight"))
                v = (w("attn_v.weight") @ n).reshape(1, 32)
                theta = pos * 5000000. ** (-np.arange(16) * 2 / 32)
                def rope(a):
                    a, b = np.split(a, 2, axis=-1)
                    return np.concatenate([a * np.cos(theta) - b * np.sin(theta), b * np.cos(theta) + a * np.sin(theta)], axis=-1).astype(np.float32)
                q, k = rope(q), rope(k)
                keys.setdefault(il, []).append(k.astype(np.float16).astype(np.float32))
                values.setdefault(il, []).append(v.astype(np.float16).astype(np.float32))
                kk, vv = np.stack(keys[il])[:, 0], np.stack(values[il])[:, 0]
                # GGML's CPU F16-K/V attention rounds Q and its online value
                # accumulator to F16. Model that numerical contract explicitly.
                score = q.astype(np.float16).astype(np.float32) @ kk.T / np.float32(np.sqrt(32))
                result = []
                for row in score:
                    high, total = -np.inf, np.float32(0)
                    acc = np.zeros(32, np.float16)
                    for z, value in zip(row, vv):
                        scale = np.float32(1)
                        if z > high:
                            scale = np.float32(np.exp(high - z)); high = z
                            acc = (acc.astype(np.float32) * scale).astype(np.float16)
                            factor = np.float32(1)
                        else: factor = np.float32(np.exp(z - high))
                        acc = (acc.astype(np.float32) + value * factor).astype(np.float16)
                        total = total * scale + factor
                    result.append(acc.astype(np.float32) / total)
                op = w("attn_output.weight") @ np.array(result).reshape(-1)
            else:
                b, c, x = np.split(w("shortconv.in_proj.weight") @ n, 3)
                bx = np.concatenate([history[il], (b * x)[:, None]], axis=1)
                history[il] = bx[:, 1:].copy()
                op = w("shortconv.out_proj.weight") @ (c * np.sum(bx * w("shortconv.conv.weight"), axis=1))
            cur = cur + op
            n = rms(cur, w("ffn_norm.weight"))
            if il == 0:
                ff = w("ffn_down.weight") @ (silu(w("ffn_gate.weight") @ n) * (w("ffn_up.weight") @ n))
            else:
                scores = w("ffn_gate_inp.weight") @ n
                if gating == 2: prob = 1 / (1 + np.exp(-scores))
                else:
                    prob = np.exp(scores - scores.max()); prob /= prob.sum()
                ids = np.argsort(-(prob + w("exp_probs_b.bias")), kind="stable")[:2]
                route = prob[ids] / max(prob[ids].sum(), 6.103515625e-5) * 1.3
                ff = sum(route[j] * (w("ffn_down_exps.weight")[e] @ (silu(w("ffn_gate_exps.weight")[e] @ n) * (w("ffn_up_exps.weight")[e] @ n))) for j, e in enumerate(ids))
            cur = cur + ff
        all_logits.append(weights["output.weight"] @ rms(cur, weights["token_embd_norm.weight"]))
    return np.array(all_logits)


@pytest.mark.parametrize("gating", [1, 2])
def test_full_mixed_model(tmp_path, gating):
    model = tmp_path / "tiny.gguf"; weights = make_model(model, gating=gating)
    tokens = [3, 17, 8, 17, 40, 1]
    expected = reference(weights, tokens, gating)
    with LiquidEngine(model, context=64, threads=2) as engine:
        actual = np.array([engine.decode([t]) for t in tokens])
        np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)
        engine.reset()
        np.testing.assert_array_equal(engine.decode(tokens), actual[-1])
        assert engine.position == len(tokens)
        before = engine.position
        with pytest.raises(ValueError): engine.decode([1, 999])
        assert engine.position == before


def test_architecture_and_payload_guards(tmp_path):
    model = tmp_path / "wrong.gguf"; make_model(model, architecture="qwen4exp")
    with pytest.raises(ValueError, match="requires architecture"): LiquidEngine(model, context=64)
    make_model(model); model.write_bytes(model.read_bytes()[:-100])
    with pytest.raises(ValueError, match="Truncated"): LiquidEngine(model, context=64)


def test_context_and_independent_handles(tmp_path):
    model = tmp_path / "tiny.gguf"; make_model(model)
    with LiquidEngine(model, context=2, threads=1) as a, LiquidEngine(model, context=2, threads=1) as b:
        a.decode([3]); b.decode([8]); b.reset()
        np.testing.assert_array_equal(a.decode([8]), b.decode([3, 8]))
        with pytest.raises(ValueError, match="capacity"): a.decode([1])
