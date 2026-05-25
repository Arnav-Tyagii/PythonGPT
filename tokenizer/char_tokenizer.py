import os
import json

class CharTokenizer:
    def __init__(self, vocab_size: int = 0):
        self.stoi = {}
        self.itos = {}
        self._vocab_size = vocab_size

    def train(self, corpus_path: str, save_dir: str):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self._vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "char_vocab.json"), "w") as f:
            json.dump(self.stoi, f)
        print(f"CharTokenizer trained. Vocab size: {self._vocab_size}")

    @classmethod
    def load(cls, save_dir: str) -> 'CharTokenizer':
        instance = cls()
        with open(os.path.join(save_dir, "char_vocab.json"), "r") as f:
            instance.stoi = json.load(f)
        instance.itos = {int(k): v for k, v in instance.stoi.items()}
        instance._vocab_size = len(instance.stoi)
        return instance

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return ''.join([self.itos.get(t, '') for t in tokens])

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
