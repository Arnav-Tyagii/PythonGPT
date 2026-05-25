import argparse
import torch
from training.config import GPTConfig, SIZE_PRESETS
from model.gpt import GPT
from tokenizer.bpe_tokenizer import BPETokenizer
from evaluation.metrics import evaluate_samples

# Load config
config = GPTConfig()
preset = SIZE_PRESETS['optimized']
for k, v in preset.items():
    setattr(config, k, v)

# Load tokenizer
tokenizer = BPETokenizer.load(config.tokenizer_path)
config.vocab_size = tokenizer.vocab_size

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate GPT model")
parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint file")
args = parser.parse_args()

# Load checkpoint
model = GPT(config)
print(f"Loading checkpoint from: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location=config.device)
model.load_state_dict(ckpt['model_state'])
model = model.to(config.device)
model.eval()
print(f"Model loaded. Params: {model.get_num_params():,}")

# Evaluate with fixed metrics
results = evaluate_samples(model, tokenizer, config, n=5)
print(f"\nSyntax rate: {results['syntax_rate']*100:.1f}%")
print(f"Valid: {results['valid']}/{results['total']}")
for s in results['samples']:
    print('-' * 40)
    print('PROMPT:', s['prompt'][:50])
    print('OUTPUT:', s['output'][:300])
    print('VALID:', s['valid'])