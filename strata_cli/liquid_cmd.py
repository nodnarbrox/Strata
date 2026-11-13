"""Explicit native Liquid CPU command; existing launch paths are unchanged."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from strata_control.liquid_native import LiquidEngine


def add_parser(sp):
    parser = sp.add_parser("liquid", help="native Liquid LFM2-MoE CPU inference (opt-in fork)")
    parser.add_argument("action", choices=("generate", "serve"))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path, help="matching Hugging Face tokenizer.json; never downloaded automatically")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--prompt")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--output", choices=("text", "json"), default="text")


class Generator:
    def __init__(self, engine, tokenizer):
        from tokenizers import Tokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from jinja2.ext import Extension
        from jinja2 import nodes
        class Generation(Extension):
            tags = {"generation"}
            def parse(self, parser):
                lineno = next(parser.stream).lineno
                body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
                return nodes.CallBlock(self.call_method("render"), [], [], body).set_lineno(lineno)
            def render(self, caller): return caller()
        self.engine = engine
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.no_truncation(); self.tokenizer.no_padding()
        vocabulary = self.tokenizer.get_vocab()
        # Check every token's ID against the GGUF, not only the vocabulary size.
        if len(vocabulary) > engine.vocab or any(engine.vocab_text(i) != text for text, i in vocabulary.items()):
            raise ValueError("Tokenizer vocabulary/IDs do not exactly match this GGUF")
        self.known_ids = set(vocabulary.values())
        self.eos = engine.meta_int("tokenizer.ggml.eos_token_id")
        self.bos = engine.meta_int("tokenizer.ggml.bos_token_id")
        self.im_end = vocabulary.get("<|im_end|>", self.eos)
        if not 0 <= self.eos < engine.vocab:
            raise ValueError("GGUF has no valid EOS token")
        template = engine.meta_string("tokenizer.chat_template")
        if not template:
            raise ValueError("GGUF has no chat template")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[Generation, "jinja2.ext.loopcontrols"])
        def fail(message): raise ValueError(str(message))
        environment.globals["raise_exception"] = fail
        self.template = environment.from_string(template)
        self.lock = threading.Lock()

    def generate(self, messages, max_tokens=1024, *, enable_thinking=True):
        import numpy as np
        if type(max_tokens) is not int or not 1 <= max_tokens <= self.engine.context:
            raise ValueError("max_tokens must fit the configured context")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        if any(not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("This port supports text system/user/assistant messages")
        prompt = self.template.render(messages=messages, add_generation_prompt=True,
                                      bos_token=self.engine.vocab_text(self.bos) or "",
                                      eos_token=self.engine.vocab_text(self.eos),
                                      enable_thinking=enable_thinking)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if len(tokens) + max_tokens > self.engine.context:
            raise ValueError("Prompt plus max_tokens exceeds the native context")
        with self.lock:
            self.engine.reset()
            start = time.perf_counter()
            logits = self.engine.decode(tokens)
            prompt_seconds = time.perf_counter() - start
            generated = []
            decode_start = time.perf_counter()
            reason = "length"
            for _ in range(max_tokens):
                token = int(np.argmax(logits))
                if token not in self.known_ids:
                    raise ValueError("Model emitted a GGUF padding ID absent from the supplied tokenizer")
                generated.append(token)
                if token in (self.eos, self.im_end):
                    reason = "stop"; break
                if len(generated) < max_tokens:
                    logits = self.engine.decode([token])
            decode_seconds = time.perf_counter() - decode_start
            seconds = time.perf_counter() - start
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        for marker in ("<|im_end|>", "<|endoftext|>"):
            if text.endswith(marker): text = text[:-len(marker)]
        reasoning = ""
        if "</think>" in text:
            reasoning, text = text.split("</think>", 1)
            reasoning = reasoning.removeprefix("<think>").strip()
        return dict(content=text.strip(), reasoning_content=reasoning, finish_reason=reason,
                    generated_ids=generated, prompt_tokens=len(tokens), completion_tokens=len(generated),
                    seconds=seconds, prompt_seconds=prompt_seconds, decode_seconds=decode_seconds,
                    generated_tok_s=len(generated) / seconds,
                    backend="strata-liquid-native-cpu", sampling="greedy", native_liquid=True,
                    timings={"prompt_n": len(tokens), "prompt_ms": prompt_seconds * 1000,
                             "prompt_per_second": len(tokens) / prompt_seconds,
                             "predicted_n": len(generated) - 1, "predicted_ms": decode_seconds * 1000,
                             "predicted_per_second": (len(generated) - 1) / decode_seconds})


def make_server(generator, port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    engine = generator.engine
    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, body):
            raw = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health": self.respond(200, {"status": "ok"})
            elif self.path == "/v1/models": self.respond(200, {"object": "list", "data": [{"id": "liquid-native", "object": "model", "owned_by": "strata"}]})
            elif self.path == "/v1/status": self.respond(200, {"service": "strata", "model": "liquid-native", "loaded": True,
                "context": {"native": engine.context}, "dialects": ["openai"], "backend": "strata-liquid-native-cpu"})
            else: self.respond(404, {"error": {"message": "Unknown endpoint"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions": return self.respond(404, {"error": {"message": "Unknown endpoint"}})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024: raise ValueError("Request size must be between 1 byte and 1 MiB")
                body = json.loads(self.rfile.read(size))
                if body.get("temperature", 0) != 0: raise ValueError("Native CPU port currently supports temperature=0 only")
                if body.get("tools") or body.get("tool_choice") or body.get("response_format") or body.get("stop"):
                    raise ValueError("Tools, constrained output and custom stop strings are not yet supported")
                if body.get("model", "liquid-native") != "liquid-native": raise ValueError("Unknown model")
                if body.get("n", 1) != 1: raise ValueError("This port supports one sequence per request")
                result = generator.generate(body["messages"], body.get("max_tokens", 1024),
                                            enable_thinking=body.get("chat_template_kwargs", {}).get("enable_thinking", True))
                message = {"role": "assistant", "content": result["content"], "reasoning_content": result["reasoning_content"]}
                usage = {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                         "total_tokens": result["prompt_tokens"] + result["completion_tokens"]}
                response = {"id": "chatcmpl-liquid-" + str(time.time_ns()), "object": "chat.completion", "created": int(time.time()),
                            "model": "liquid-native", "choices": [{"index": 0, "message": message, "finish_reason": result["finish_reason"]}], "usage": usage}
                if result.get("timings"): response["timings"] = result["timings"]
                if not body.get("stream"):
                    return self.respond(200, response)
                # A single buffered SSE delta preserves the host protocol. Token
                # streaming and cancellation are not claimed by this CPU port.
                self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close"); self.end_headers()
                chunk = {k: v for k, v in response.items() if k not in ("choices", "usage")}
                chunk["object"] = "chat.completion.chunk"
                delta = dict(message)
                chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": None}]
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": result["finish_reason"]}]; chunk["usage"] = usage
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
                self.close_connection = True
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(generator, port):
    server = make_server(generator, port)
    print(json.dumps({"ready": True, "url": f"http://127.0.0.1:{server.server_port}/v1", "backend": "strata-liquid-native-cpu"}), flush=True)
    try: server.serve_forever()
    finally: server.server_close()


def run(args):
    with LiquidEngine(args.model_path, library=args.library, context=args.context, threads=args.threads) as engine:
        generator = Generator(engine, args.tokenizer)
        if args.action == "serve": serve(generator, args.port); return 0
        if args.prompt is None: raise ValueError("generate requires --prompt")
        result = generator.generate([{"role": "user", "content": args.prompt}], args.max_tokens)
        print(json.dumps(result, ensure_ascii=False) if args.output == "json" else result["content"])
        return 0
