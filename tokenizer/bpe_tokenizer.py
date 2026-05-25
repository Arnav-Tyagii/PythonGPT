import os
from tokenizers import ByteLevelBPETokenizer

class BPETokenizer:
    def __init__(self, vocab_size: int = 8000):
        self._vocab_size = vocab_size
        self._tokenizer = None

    def train(self, corpus_path: str, save_dir: str):
        tok = ByteLevelBPETokenizer()
        tok.train(
            files=[corpus_path],
            vocab_size=self._vocab_size,
            min_frequency=2,
            special_tokens=["<|endoftext|>", "<|pad|>", "<|comment|>", "<|docstring|>"]
        )
        os.makedirs(save_dir, exist_ok=True)
        tok.save_model(save_dir)
        self._tokenizer = tok
        print(f"Tokenizer trained. Vocab size: {tok.get_vocab_size()}")

    @classmethod
    def load(cls, save_dir: str) -> 'BPETokenizer':
        instance = cls()
        instance._tokenizer = ByteLevelBPETokenizer.from_file(
            os.path.join(save_dir, "vocab.json"),
            os.path.join(save_dir, "merges.txt")
        )
        instance._vocab_size = instance._tokenizer.get_vocab_size()
        return instance

    def encode(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")
        return self._tokenizer.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")
        return self._tokenizer.decode(tokens)

    @property
    def vocab_size(self) -> int:
        if self._tokenizer:
            return self._tokenizer.get_vocab_size()
        return self._vocab_size
