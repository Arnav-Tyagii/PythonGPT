import os
import sys
import argparse
import subprocess
import torch

from training.config import GPTConfig, SIZE_PRESETS
from tokenizer.bpe_tokenizer import BPETokenizer
from training.dataset import get_dataloaders
from model.gpt import GPT
from training.trainer import Trainer
from evaluation.metrics import evaluate_samples

def run_script(script_path, args=None):
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT on algorithm data")
    parser.add_argument("--algo-only", action="store_true", help="Use only algorithm data (no mixing)")
    parser.add_argument("--repeat", type=int, default=20, help="How many times to repeat algo data")
    parser.add_argument("--iters", type=int, default=3000, help="Fine-tuning iterations")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default 5e-5)")
    parser.add_argument("--resume", action="store_true", help="Resume from finetuned_model.pt")
    args = parser.parse_args()

    # Step 2: Download algorithm data if not exists
    algo_data_path = "data/algorithms_corpus.txt"
    if not os.path.exists(algo_data_path):
        print(f"{algo_data_path} not found. Downloading algorithms dataset...")
        run_script("data/download_algorithms.py")

    # Step 3: Prepare mixed dataset
    mix_data_path = "data/finetune_corpus.txt"
    if args.algo_only:
        print("Using algorithm data only (no mixing). WARNING: may cause catastrophic forgetting.")
        config_data_path = algo_data_path
    else:
        print(f"Preparing mixed dataset (repeat={args.repeat})...")
        run_script("data/finetune_prepare.py")
        config_data_path = mix_data_path

    # Step 4-6: Setup Config, Tokenizer, Model
    config = GPTConfig()
    
    # Load preset
    preset = SIZE_PRESETS["optimized"]
    for k, v in preset.items():
        setattr(config, k, v)
        
    config.data_path = config_data_path
    
    tokenizer = BPETokenizer.load(config.tokenizer_path)
    config.vocab_size = tokenizer.vocab_size
    
    model = GPT(config)
    ckpt_path = 'checkpoints/best_model.pt'
    start_iter = 0
    is_resuming = False
    
    if args.resume and os.path.exists('checkpoints/finetuned_model.pt'):
        print("Resuming fine-tuning from checkpoints/finetuned_model.pt...")
        ckpt_path = 'checkpoints/finetuned_model.pt'
        is_resuming = True
    else:
        if args.resume:
            print("finetuned_model.pt not found. Falling back to base model...")
        print("Loading base model from checkpoints/best_model.pt...")
        if not os.path.exists(ckpt_path):
            print(f"Error: {ckpt_path} not found. Cannot fine-tune without a base model.")
            sys.exit(1)
            
    ckpt = torch.load(ckpt_path, map_location=config.device)
    model.load_state_dict(ckpt['model_state'])
    model.to(config.device)
    
    if is_resuming:
        start_iter = ckpt.get('iter', 0) + 1
    
    # --- VRAM DRY RUN TESTER ---
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
                # Using updated torch.amp.GradScaler to avoid the deprecation warning you saw earlier
                dummy_scaler = torch.amp.GradScaler('cuda')
                
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
    
    # Step 7: Create DataLoaders
    print("Building DataLoaders for fine-tuning...")
    train_loader, val_loader = get_dataloaders(config, tokenizer)

    # Step 8: Apply Finetuning specific hyperparameters
    config.learning_rate = args.lr
    config.min_lr = 5e-6
    config.warmup_iters = 100
    config.max_iters = args.iters
    config.eval_interval = 200
    config.grad_clip = 0.5
    
    # We will override the Trainer's save mechanism to save to finetuned_model.pt
    # The Trainer usually saves to checkpoints/best_model.pt. 
    # We can either monkey-patch the trainer's save_checkpoint, or modify Trainer to accept a save name.
    # To keep it simple without touching trainer.py if we don't have to, let's just let it run 
    # Wait, the prompt says: "Save checkpoints to checkpoints/finetuned_model.pt (do NOT overwrite best_model.pt)".
    # Let's check if Trainer accepts a checkpoint name, or if we should modify trainer.py.
    # I will modify config or monkey patch trainer.
    
    # Simple monkey patch to prevent overwriting best_model.pt
    trainer = Trainer(model, config, train_loader, val_loader, tokenizer)
    
    if is_resuming and 'optimizer_state' in ckpt:
        trainer.optimizer.load_state_dict(ckpt['optimizer_state'])
        trainer.best_val = ckpt.get('val_loss', float('inf'))
        print(f"Restored optimizer state. Starting from iteration {start_iter} with best val_loss: {trainer.best_val:.4f}")
    
    # Monkey patch save_checkpoint
    original_save = trainer.save_checkpoint
    def safe_save_checkpoint(step, val_loss):
        # We temporarily change the behavior
        import torch
        state = {
            'model_state': trainer.model.state_dict(),
            'optimizer_state': trainer.optimizer.state_dict(),
            'iter': step,
            'val_loss': val_loss,
            'config': trainer.config
        }
        os.makedirs(trainer.config.checkpoint_dir, exist_ok=True)
        # Always save to finetuned_model.pt
        save_path = os.path.join(trainer.config.checkpoint_dir, 'finetuned_model.pt')
        torch.save(state, save_path)
        print(f"--> Saved fine-tuned checkpoint to {save_path} (step: {step}, val_loss: {val_loss:.4f})")
    
    trainer.save_checkpoint = safe_save_checkpoint

    # Step 9: Train
    print(f"\nStarting Fine-tuning for {config.max_iters} iterations with LR={config.learning_rate}...")
    trainer.train(start_iter=start_iter)

    # Step 10: Evaluate
    print("\nFine-tuning complete. Running post-training evaluation...")
    model.eval()
    eval_results = evaluate_samples(model, tokenizer, config)
    print(f"Syntax Rate: {eval_results['syntax_rate']*100:.1f}%")

    print("\nSample generations (Fine-tuned):")
    for sample in eval_results['samples'][:3]:
        print("-" * 40)
        print("PROMPT:")
        print(sample['prompt'])
        print("OUTPUT:")
        print(sample['output'])
        print(f"VALID: {sample['valid']}")
    print("-" * 40)
    print("Fine-tuning pipeline complete. Use the new finetuned_model.pt in app.py!")

if __name__ == "__main__":
    main()
