from dataclasses import dataclass, field
import torch

@dataclass
class GPTConfig:
    # Optimized config for RTX 3050 Ti: params ≈ 12 * 10 * 640^2 ≈ 49,152,000 parameters
    # Architecture
    vocab_size: int = 8000
    block_size: int = 512
    n_embd: int = 640
    n_head: int = 8
    n_layer: int = 10
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    gradient_accumulation_steps: int = 8   # effective batch = 128
    max_iters: int = 30000
    eval_interval: int = 500
    eval_iters: int = 100
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Sampling defaults
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95

    # Paths + system
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    data_path: str = "data/python_corpus.txt"
    tokenizer_path: str = "tokenizer/vocab"
    compile_model: bool = False
    tokenizer_type: str = "bpe"

SIZE_PRESETS = {
    "nano":   dict(n_layer=4,  n_head=4,  n_embd=256),   # ~3M params
    "small":  dict(n_layer=8,  n_head=8,  n_embd=512),   # ~25M params (default)
    "medium": dict(n_layer=12, n_head=12, n_embd=768),   # ~85M params
    "optimized": dict(n_layer=10, n_head=8, n_embd=640), # ~49M params
}
