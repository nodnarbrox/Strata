"""Whole-prompt native-port qualification, with a separate reference process.

Reference is the caller-supplied opaque harness bridge, used only by this test.
The shipped Liquid engine does not link it. Run phases sequentially to avoid
overlapping model owners. Raw logits, prompts, IDs and text remain in --out.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

TASKS = [
    ("arithmetic", "What is 17 multiplied by 19? Give the final answer as just the integer.", "323"),
    ("logic", "All dax are wugs. No wugs are red. Can a dax be red? Give a final answer of just Yes or No.", "no"),
    ("instruction", "Reply with exactly the word orchid in your final answer.", "orchid"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("prepare", "reference", "native", "compare"))
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bridge", type=Path)
    ap.add_argument("--reference-module", type=Path)
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    if args.phase == "prepare":
        from strata_control.liquid_native import LiquidEngine
        from strata_cli.liquid_cmd import Generator
        with LiquidEngine(args.model, threads=6) as e:
            gen = Generator(e, args.tokenizer)
            rows = []
            for name, prompt, _ in TASKS:
                rendered = gen.template.render(messages=[{"role": "user", "content": prompt}],
                    add_generation_prompt=True, bos_token=e.vocab_text(gen.bos), eos_token=e.vocab_text(gen.eos), enable_thinking=True)
                rows.append(dict(task=name, prompt=prompt, rendered=rendered, ids=gen.tokenizer.encode(rendered, add_special_tokens=False).ids))
            meta = dict(vocab=e.vocab, eos=gen.eos, im_end=gen.im_end, rows=rows,
                model_path=str(args.model), model_bytes=args.model.stat().st_size,
                tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest())
            (args.out / "input.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps({"prepared": True, "vocab": meta["vocab"], "prompt_tokens": [len(r["ids"]) for r in rows]}), flush=True)
        return
    inputs = json.loads((args.out / "input.json").read_text())
    if args.phase == "compare":
        a = json.loads((args.out / "reference.json").read_text())
        b = json.loads((args.out / "native.json").read_text())
        rows = []
        for i, (ra, rb) in enumerate(zip(a["rows"], b["rows"])):
            aa, bb = np.load(args.out / f"reference-{i}.npy"), np.load(args.out / f"native-{i}.npy")
            # Saved logits cover the same complete prompt at every input position.
            diff = np.abs(aa - bb)
            rows.append(dict(task=ra["task"], prompt_max_abs_logit_error=float(diff.max()),
                prompt_mean_abs_logit_error=float(diff.mean()), prompt_argmax_match=int((aa.argmax(1) == bb.argmax(1)).sum()),
                prompt_positions=len(aa), generated_ids_equal=ra["generated_ids"] == rb["generated_ids"],
                reference_correct=ra["correct"], native_correct=rb["correct"], native_finish=rb["finish_reason"]))
        result = dict(complete=True, rows=rows, native_generated_tok_s=b["generated_tok_s"],
                      reference_generated_tok_s=a["generated_tok_s"], native_correct=sum(r["native_correct"] for r in rows),
                      native_naturally_stopped=sum(r["native_finish"] == "stop" for r in rows),
                      topology="i9-9900K CPU, six threads, one sequence, 4096 context, F16 K/V",
                      scope="Pristine GGUF architecture parity and three smoke tasks; not broad quality or a speedup claim")
        (args.out / "comparison.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    if args.phase == "reference":
        spec = importlib.util.spec_from_file_location("reference_bridge", args.reference_module)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        e = module.NativeBatch(args.bridge, args.model, n_seq=1, per_seq_ctx=4096, n_threads=6, gpu_layers=0)
        def step(token, pos): return e._step([(0, token, pos, 1)])[0]
        for row in inputs["rows"]:
            if e.tok(row["rendered"]) != row["ids"]: raise RuntimeError("Reference tokenizer IDs differ")
    else:
        from strata_control.liquid_native import LiquidEngine
        e = LiquidEngine(args.model, threads=6)
        def step(token, pos): return e.decode([token])
    rows = []
    try:
        for i, row in enumerate(inputs["rows"]):
            e.reset(); start = time.perf_counter(); prompt_logits = []
            for pos, token in enumerate(row["ids"]):
                logits = step(token, pos); prompt_logits.append(logits.copy())
            prompt_seconds = time.perf_counter() - start
            np.save(args.out / f"{args.phase}-{i}.npy", np.array(prompt_logits))
            ids = []; stop = "length"; decode_start = time.perf_counter()
            for j in range(1024):
                token = int(np.argmax(logits)); ids.append(token)
                if token in (inputs["eos"], inputs["im_end"]): stop = "stop"; break
                logits = step(token, len(row["ids"]) + j)
            seconds = time.perf_counter() - start
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(args.tokenizer))
            full = tokenizer.decode(ids, skip_special_tokens=False)
            content = full.split("</think>")[-1].replace("<|im_end|>", "").strip()
            correct = stop == "stop" and content.lower().rstrip(".!") == TASKS[i][2]
            result = dict(task=row["task"], generated_ids=ids, full=full, content=content, correct=correct,
                          finish_reason=stop, tokens=len(ids), seconds=seconds, prompt_seconds=prompt_seconds,
                          decode_seconds=time.perf_counter() - decode_start)
            rows.append(result)
            report = dict(phase=args.phase, complete=len(rows) == len(TASKS), rows=rows,
                          generated_tok_s=sum(r["tokens"] for r in rows) / sum(r["seconds"] for r in rows))
            (args.out / f"{args.phase}.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in result.items() if k not in ("full", "generated_ids")}), flush=True)
    finally: e.close()


if __name__ == "__main__": main()
