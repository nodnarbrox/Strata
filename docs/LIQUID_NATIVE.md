# Native Liquid CPU support in this fork

Status: experimental CPU architecture support. Real generation works, but this
port has not qualified transparent parity with the pinned reference engine.

This fork adds an explicit `strata liquid` command for LiquidAI
LFM2.5-8B-A1B GGUF (`general.architecture = lfm2moe`). Existing Strata launch
commands and the Qwen/DeepSeek engine paths are unchanged.

The new C++ graph executes the model with Strata's vendored GGML CPU kernels.
It does not link llama.cpp, call `llama_decode`, launch `llama-server`, or
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
