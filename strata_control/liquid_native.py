"""Strata's opt-in native CPU Liquid engine. Token IDs in, unmodified logits out.

This module has no tokenizer, answer lookup, sampling callback or server fallback.
The C++ engine owns every model forward and the recurrent/KV state.
"""
from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
import sys


class LiquidEngine:
    def __init__(self, model, *, library=None, context=4096, threads=6):
        root = Path(__file__).resolve().parents[1]
        name = "strata-liquid.dll" if sys.platform == "win32" else "libstrata-liquid.so"
        library = Path(library or os.environ.get("STRATA_LIQUID_LIBRARY", root / "build-liquid" / name)).resolve()
        if not library.is_file():
            raise ValueError(f"Native Liquid engine missing: {library}. Build with cmake -S src/liquid -B build-liquid.")
        self.lib = ct.CDLL(str(library))
        self.lib.sl_error.restype = ct.c_char_p
        self.lib.sl_create.argtypes = [ct.c_char_p, ct.c_int, ct.c_int]
        self.lib.sl_create.restype = ct.c_void_p
        self.lib.sl_free.argtypes = [ct.c_void_p]
        for method in ("sl_vocab", "sl_position", "sl_reset"):
            getattr(self.lib, method).argtypes = [ct.c_void_p]
            getattr(self.lib, method).restype = ct.c_int
        self.lib.sl_decode.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int32), ct.c_int, ct.POINTER(ct.c_float), ct.c_int]
        self.lib.sl_decode.restype = ct.c_int
        self.lib.sl_meta_string.argtypes = [ct.c_void_p, ct.c_char_p]
        self.lib.sl_meta_string.restype = ct.c_char_p
        self.lib.sl_meta_int.argtypes = [ct.c_void_p, ct.c_char_p]
        self.lib.sl_meta_int.restype = ct.c_int
        self.lib.sl_vocab_text.argtypes = [ct.c_void_p, ct.c_int]
        self.lib.sl_vocab_text.restype = ct.c_char_p
        self.handle = self.lib.sl_create(os.fsencode(Path(model).resolve()), context, threads)
        if not self.handle:
            raise ValueError(self.lib.sl_error().decode("utf-8", errors="replace"))
        self.vocab = self.lib.sl_vocab(self.handle)
        self._logits = (ct.c_float * self.vocab)()
        self.context = context

    def decode(self, tokens):
        import numpy as np
        if not self.handle:
            raise ValueError("Liquid engine is closed")
        values = list(tokens)
        if not values or any(type(x) is not int or not 0 <= x < self.vocab for x in values):
            raise ValueError("Decode requires nonempty integer token IDs inside the vocabulary")
        ids = (ct.c_int32 * len(values))(*values)
        if self.lib.sl_decode(self.handle, ids, len(values), self._logits, self.vocab):
            raise ValueError(self.lib.sl_error().decode("utf-8", errors="replace"))
        return np.ctypeslib.as_array(self._logits).copy()

    @property
    def position(self):
        return self.lib.sl_position(self.handle)

    def meta_string(self, key):
        value = self.lib.sl_meta_string(self.handle, key.encode())
        return value.decode("utf-8") if value is not None else None

    def meta_int(self, key):
        return self.lib.sl_meta_int(self.handle, key.encode())

    def vocab_text(self, token):
        value = self.lib.sl_vocab_text(self.handle, token)
        return value.decode("utf-8") if value is not None else None

    def reset(self):
        if self.lib.sl_reset(self.handle):
            raise ValueError(self.lib.sl_error().decode())

    def close(self):
        if getattr(self, "handle", None):
            self.lib.sl_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
