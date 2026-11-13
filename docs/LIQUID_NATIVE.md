# Native Liquid CPU support in this fork

Status: experimental CPU architecture support. An opt-in Linux build using
pinned reference GGML kernels and matching cache padding passed five complete
reference generation comparisons. This is bounded parity evidence; the default
vendored build and other configurations do not inherit that result.

This fork adds an explicit `strata liquid` command for LiquidAI
LFM2.5-8B-A1B GGUF (`general.architecture = lfm2moe`). Existing Strata launch
commands and the Qwen/DeepSeek engine paths are unchanged.

The new C++ graph executes the model with the selected GGML CPU kernels.
It does not link `libllama`, call `llama_decode`, launch `llama-server`, or
discover another inference server. Hugging Face Tokenizers only encodes and
decodes text; Jinja renders the actual template stored in the GGUF.

The native implementation includes operator RMS normalization, gated causal
short convolution (the history is B*X), per-head Q/K normalization, NEOX RoPE,
GQA with F16 K/V, dense SwiGLU and sigmoid/softmax MoE with selection bias and
normalized top-k weights. Convolution and attention state belong to each engine
handle and reset together between requests. Architecture metadata and tensor
shapes are checked before execution. Unsupported model variants are refused.

## Build and run

Requires CMake 3.24+, a C++20 compiler and the sources in this repository.
The standalone build only builds this CPU engine; it does not require CUDA.
Build on the machine that will run the library, because GGML enables its native
CPU instructions. On Windows, use the Visual Studio x64 developer environment.

```sh
cmake -S src/liquid -B build-liquid -DCMAKE_BUILD_TYPE=Release
cmake --build build-liquid --config Release -j 2
python -m pip install numpy tokenizers jinja2
```

For the separately qualified Linux reference-kernel path, supply matching GGML
headers, CPU library and base library explicitly. All three are required. This
uses the existing libraries without downloading them or building an inference
server. Keep the complete compatible runtime bundle available at execution time.

```sh
cmake -S src/liquid -B build-liquid-reference -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_LIQUID_GGML_INCLUDE_DIR=/path/reference/include \
  -DSTRATA_LIQUID_GGML_CPU_LIBRARY=/path/reference/runtime/libggml-cpu-haswell.so \
  -DSTRATA_LIQUID_GGML_BASE_LIBRARY=/path/reference/runtime/libggml-base.so
cmake --build build-liquid-reference -j 2
```

Select `build-liquid-reference/libstrata-liquid.so` using `--library` or
`STRATA_LIQUID_LIBRARY`. Headers and shared libraries must be from the same
build; arbitrary versions are not qualified. The checked bundle is GGML0.28
from reference commit `8642541b255bfb5dc39cd351a178be09013a55d6`:
CPU SHA256 `5bd145bd299d66c80a766252659b20d25b5300090089aa542862388a3c219aa6`,
base SHA256 `50d78f64f3242d5f5618686f3239ceea0bb5e49310cd7eef8789830d3cb853de`.
If that bundle requires a newer system ABI, use its compatible loader and
libraries as a unit. Only Linux external linkage has been checked.

This path also rounds the visible K/V cache length to256-token blocks and
masks unused positions. The compact and padded CPU attention reductions start
diverging at257 keys in the checked configuration; using the same arithmetic
geometry prevents those differences from accumulating. The setting applies to
every prompt, with no question-specific branches or answer-token controls.

With a single-configuration generator such as Ninja, the output is
`build-liquid/libstrata-liquid.so` (Linux) or `build-liquid/strata-liquid.dll`
(Windows). For a Visual Studio multi-configuration build, supply
`--library build-liquid/Release/strata-liquid.dll` when running the commands.

Use the existing model file and its matching `tokenizer.json`. Nothing is
downloaded automatically. Every provided vocabulary entry must have the same
ID and text in the GGUF. Extra GGUF padding IDs may remain unencodable; emitting
one is an error, never a silently dropped token. The qualified tokenizer is
from the official LiquidAI repository at revision
`feb5e04da4910dd56d33f0cd03747ff298c1e801`, SHA256
`695be7802a0e4b8a81048f0ff5ebb7fc811a0ba5a6be63dbb24deb5a81096f41`.

```sh
python strata.py liquid generate --model-path /path/model.gguf --tokenizer /path/tokenizer.json --prompt "Explain binary search." --output json
python strata.py liquid serve --model-path /path/model.gguf --tokenizer /path/tokenizer.json --port 8096
```

The worker binds to `127.0.0.1` and exposes `/health`, `/v1/status`,
`/v1/models` and `/v1/chat/completions`. Its model ID is `liquid-native`.
Requests use `temperature: 0`. Set `STRATA_SERVER_URL=http://127.0.0.1:8096`
in the existing Strata host to select this worker through its normal identity
probe. Its reported backend is `strata-liquid-native-cpu`.

For code that already owns tokenization:

```python
from strata_control.liquid_native import LiquidEngine

with LiquidEngine("model.gguf", context=4096, threads=6) as model:
    logits = model.decode(prompt_token_ids)
    next_token = int(logits.argmax())
    logits = model.decode([next_token])
    model.reset()
```

The returned logits are copied, so later decodes cannot mutate previous results.
An engine handle must have one caller at a time. The HTTP server serializes
generation using a lock. It accepts simultaneous connections but executes one
sequence at a time.

## Supported scope

The initial port targets single-file, causal LFM2-MoE with default RoPE, F32
convolution kernels and the geometry carried by the GGUF. The production model
qualification uses LFM2.5-8B-A1B, 24 layers, two leading dense layers, 32 experts
and four selected experts. Other compatible-looking checkpoints need their own
qualification.

GPU offload, split GGUF, sliding-window attention, scaled/partial RoPE,
LoRA adapters, non-greedy sampling, tool calls, constrained output and custom
stop strings are not implemented. Edited single-file GGUF weights can be loaded,
but edits still need fresh quality tests. No runtime answer heads or token-edit
callbacks are part of this port.

The CPU graph processes prompt tokens sequentially. It prioritizes a verifiable
architecture implementation; no throughput improvement over llama.cpp is
claimed. SSE responses are buffered until generation completes. Incremental
token streaming, cancellation and optimized batched prefill remain future work.

## Validation

```sh
python -m pytest tests/liquid -q
```

The independent NumPy oracle runs a complete synthetic mixed model, checking
both routing functions, selection bias, expert scaling, causal convolution
history, GQA/RoPE and state reset. Its attention oracle explicitly accounts for
the CPU kernel's F16 query and value-accumulator rounding. Separate checks cover
independent handles, context limits, wrong architectures and truncated tensors.
HTTP tests cover discovery, JSON/SSE replies and refusal of unsupported controls.

The Windows and Linux native builds both passed the four architecture/state
tests. Six HTTP tests passed. Four applicable Terminal import tests passed and
one was skipped by its existing conditions.

The fresh CPU model check produced correct, naturally stopping answers for all
three arithmetic, logic and exact-word questions. Through the real Strata host,
307 generated tokens took 20.2918 request seconds: **15.1293 tokens/second**.
The final timing-report check repeated arithmetic after adding measured server
timings: 61 tokens in 4.9102 seconds, **12.4232 tokens/second** for that complete
request and 18.6185 native decode tokens/second. These runs used an i9-9900K CPU,
six threads, one sequence, context 4096 and F16 K/V. Variation and different
generated streams prevent claiming a speedup over the reference.

The initial direct native/reference comparison passed all three final-answer
checks but did not preserve complete generated IDs. Identical prompt positions
had maximum logit differences up to 9.4298. Read-only diagnostics found all 24
layer outputs exactly equal for the first token; later attention differences
began small and amplified through the quantized graph. This does not establish
their sole cause or broad output quality. The independent synthetic oracle
passed, but it cannot establish production-checkpoint equivalence. Use the
reference path for work that requires previously qualified exact behavior.

The opt-in pinned-kernel follow-up preserved the native graph and checkpoint.
Kernel replacement alone matched27 prompt positions and four whole replies,
but a copying reply first differed at decision position319. Common-prefix
numerical diagnosis localized its first logit drift to position256, beginning
in the first attention layer. A synthetic operator test reproduced compact
versus padded differences above256 keys with the same kernel library.

Matching that cache geometry then gave zero logit error across all320 replay
positions and passed a fresh five-task reference/native/reference generation
trial: arithmetic, logic, instruction following, the original wrong code reply
and a longer copying reply. Every generated ID and natural stop matched. The
known wrong answer remains wrong: this qualifies the checked engine behavior,
not new reasoning ability. Teacher-forced replay is diagnostic evidence only;
the generation comparisons use actual greedy logits without supplied answers.

On an i5-9500 with six physical CPU cores, one sequence and4096 context, the
trial generated1385 tokens per arm. Reference took62.2904 seconds
(22.2346 tokens/second), native65.6193 (21.1066), repeat reference62.5265
(22.1506). Timing includes sequential prompt processing and decoding, excluding
loading. The native path was slower on this panel. No100x speedup, GPU support,
LoRA support or broad checkpoint equivalence follows from these five tests.

The packaged CMake build subsequently passed all10 existing independent
synthetic/HTTP checks and all five fresh complete generation comparisons against
those reference repeats. Its measured baseline was1385 tokens in65.8552 seconds,
**21.0310 generated tokens/second**. Invalid partial paths, missing headers and
missing shared libraries were rejected during configuration. These checks use
the pinned Linux CPU bundle; external Windows linkage remains unsupported.

The SSE endpoint carries measured prompt/decode timings. Its first answer token
comes from prompt processing, so `predicted_n` counts subsequent decode forwards.
The host must use these timings instead of interpreting buffered delivery as
the model's generation rate.

`tools/qualify_liquid_native.py` records real whole-prompt logits and naturally
stopping generations. Its optional reference phase uses a separately supplied
opaque harness bridge in another process, only for qualification. Save each
phase's raw artifacts and run phases sequentially; do not run multiple owners
of the model while measuring performance. The model path, tokenizer identity,
prompt IDs, logits and full generated IDs/text are retained in the output.

Architecture references:

- [Official LiquidAI model configuration](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/blob/main/config.json)
- [Pinned reference architecture and tensors](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/src/models/lfm2moe.cpp)
- [Pinned reference short convolution](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/src/models/lfm-base.cpp)
- [Official tokenizer update](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/commit/feb5e04da4910dd56d33f0cd03747ff298c1e801)
