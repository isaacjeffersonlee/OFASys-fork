# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from .gpt2_bpe_utils import get_encoder
import os

OFA_CACHE_HOME = os.getenv("OFA_CACHE_HOME")
if OFA_CACHE_HOME in [None, "", " "]:
    raise EnvironmentError(
        f"Environment variable {OFA_CACHE_HOME} is not bound, but should have been set by pipeline.py"
    )
DEFAULT_ENCODER_JSON = os.path.join(OFA_CACHE_HOME, "encoder.json")
DEFAULT_VOCAB_BPE = os.path.join(OFA_CACHE_HOME, "vocab.bpe")
DEFAULT_DICT_BPE = os.path.join(OFA_CACHE_HOME, "dict.txt")


class GPT2BPE(object):
    def __init__(self):
        encoder_json = DEFAULT_ENCODER_JSON
        vocab_bpe = DEFAULT_VOCAB_BPE
        self.bpe = get_encoder(encoder_json, vocab_bpe)

    def encode(self, x: str) -> str:
        return " ".join(map(str, self.bpe.encode(x)))

    def decode(self, x: str) -> str:
        return self.bpe.decode(
            [
                int(tok)
                if tok not in {"<unk>", "<mask>"} and not tok.startswith("<")
                else tok
                for tok in x.split()
            ]
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(" ")

    def _encode(self, x: str):
        return self.bpe.encode(x)

    @property
    def eod(self):
        return self.bpe.encoder["<|endoftext|>"]

    @property
    def vocab_size(self):
        return len(self.bpe.encoder)
