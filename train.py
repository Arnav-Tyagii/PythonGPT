import os
import sys
import argparse
import torch
from training.config import GPTConfig, SIZE_PRESETS
from tokenizer.bpe_tokenizer import BPETokenizer
from training.dataset import get_dataloaders
from model.gpt import GPT
from training.trainer import Trainer
from evaluation.metrics import compute_perplexity, evaluate_samples

def main():
    parser = argparse.ArgumentParser(description="Train GPT on Python code")
    parser.add_argument("--size", type=str, choices=["nano", "small", "medium", "optimized"], default="small", help="Model size preset")
    parser.add_argument("--iters", type=int, help="Override max_iters")
    parser.add_argument("--compile", action="store_true", help="Compile model (requires PyTorch 2.0+)")
    parser.add_argument("--resume", action="store_true", help="Resume from best_model.pt")
    args = parser.parse_args()

    config = GPTConfig()
    
    if args.size in SIZE_PRESETS:
        for k, v in SIZE_PRESETS[args.size].items():
            setattr(config, k, v)
            
    if args.iters:
        config.max_iters = args.iters
        
    if args.compile:
        config.compile_model = True

    if not os.path.exists(config.data_path):
        print(f"Error: {config.data_path} not found. Run python prepare_data.py first.")
        sys.exit(1)

    if not os.path.exists(os.path.join(config.tokenizer_path, "vocab.json")):
        print(f"Error: Tokenizer not found at {config.tokenizer_path}. Run python prepare_data.py first.")
        sys.exit(1)

    tokenizer = BPETokenizer.load(config.tokenizer_path)
    config.vocab_size = tokenizer.vocab_size

    print("Creating model...")
    model = GPT(config)
    model.to(config.device)

    if config.device == 'cuda':
        print("\n--- Running VRAM Dry Run ---")
        
        while config.batch_size >= 1:
            try:
                torch.cuda.reset_peak_memory_stats()
                
                # Dummy inputs
                dummy_x = torch.randint(0, config.vocab_size, (config.batch_size, config.block_size), device=config.device)
                dummy_y = torch.randint(0, config.vocab_size, (config.batch_size, config.block_size), device=config.device)
                
                # Dummy optimizer
                dummy_opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
                dummy_scaler = torch.cuda.amp.GradScaler()
                
                dummy_opt.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _, loss = model(dummy_x, dummy_y)
                    
                dummy_scaler.scale(loss).backward()
                dummy_scaler.step(dummy_opt)
                
                peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                
                # Cleanup dummy memory
                del dummy_x, dummy_y, loss, dummy_opt, dummy_scaler
                torch.cuda.empty_cache()
                
                print(f"Testing batch_size={config.batch_size}: Peak VRAM usage = {peak_vram:.2f} GB")
                
                if peak_vram <= 3.5:
                    print(f"VRAM Check Passed! Effective config: batch_size={config.batch_size}, grad_accum={config.gradient_accumulation_steps}")
                    break
                else:
                    if config.batch_size == 1:
                        print("WARNING: Even batch_size=1 exceeds 3.5GB! Proceeding anyway as last resort...")
                        break
                    print("WARNING: VRAM exceeds 3.5GB! Halving batch size...")
                    config.batch_size //= 2
                    config.gradient_accumulation_steps *= 2
                    
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"Testing batch_size={config.batch_size}: OOM Error! Halving batch size...")
                if config.batch_size == 1:
                    print("WARNING: Even batch_size=1 hit OOM! Proceeding anyway as last resort...")
                    break
                config.batch_size //= 2
                config.gradient_accumulation_steps *= 2
                
        print("----------------------------\n")

    print("Building DataLoaders...")
    train_loader, val_loader = get_dataloaders(config, tokenizer)
    
    param_count = model.get_num_params()
    size_mb = param_count * 4 / (1024 * 1024)
    print(f"Model parameters: {param_count:,} ({size_mb:.2f} MB)")
    print(f"--> Using device: {config.device.upper()} <--\n")

    if config.compile_model:
        print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")

    trainer = Trainer(model, config, train_loader, val_loader, tokenizer)
    start_iter = 0
    if args.resume:
        start_iter = trainer.load_checkpoint()
    trainer.train(start_iter=start_iter)

    print("\nTraining complete. Running post-training evaluation...")
    perplexity = compute_perplexity(model, val_loader, config)
    print(f"Validation Perplexity: {perplexity:.4f}")

    print("\nEvaluating sample syntax validity...")
    eval_results = evaluate_samples(model, tokenizer, config)
    print(f"Syntax Rate: {eval_results['syntax_rate']*100:.1f}%")

    print("\nSample generations:")
    for sample in eval_results['samples'][:3]:
        print("-" * 40)
        print("PROMPT:")
        print(sample['prompt'])
        print("OUTPUT:")
        print(sample['output'])
        print(f"VALID: {sample['valid']}")
    print("-" * 40)

if __name__ == "__main__":
    main()
