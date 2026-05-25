import os
import math
import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, tokenizer):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.best_val = float('inf')
        self.patience_counter = 0

        # Selective weight decay: 2D params only (weight matrices)
        # Biases and LayerNorm params get NO weight decay
        decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
        no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
        self.optimizer = torch.optim.AdamW([
            {'params': decay, 'weight_decay': config.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ], lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.device == 'cuda'))

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        self.log_path = os.path.join(config.log_dir, 'loss_log.csv')
        with open(self.log_path, 'w') as f:
            f.write('iter,train_loss,val_loss,lr\n')

    def get_lr(self, iter):
        if iter < self.config.warmup_iters:
            return self.config.learning_rate * iter / self.config.warmup_iters
        if iter > self.config.max_iters:
            return self.config.min_lr
        progress = (iter - self.config.warmup_iters) / (self.config.max_iters - self.config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        out = {}
        for split, loader in [('train', self.train_loader), ('val', self.val_loader)]:
            losses = []
            for i, (x, y) in enumerate(loader):
                if i >= self.config.eval_iters:
                    break
                x, y = x.to(self.config.device), y.to(self.config.device)
                with torch.autocast(device_type='cuda' if self.config.device == 'cuda' else 'cpu', dtype=torch.float16, enabled=(self.config.device == 'cuda')):
                    _, loss = self.model(x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses) if losses else 0.0
        self.model.train()
        return out

    def load_checkpoint(self):
        path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.best_val = checkpoint['val_loss']
            print(f"Resumed from iter {checkpoint['iter']} with val_loss {self.best_val:.4f}")
            return checkpoint['iter'] + 1
        print("No checkpoint found. Starting from scratch.")
        return 0

    def save_checkpoint(self, iter, val_loss):
        path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'iter': iter,
            'val_loss': val_loss,
        }, path)
        print(f"  Saved checkpoint  val_loss={val_loss:.4f}")

    def train(self, start_iter=0):
        self.model.train()
        train_iter = iter(self.train_loader)
        pbar = tqdm(range(start_iter, self.config.max_iters), desc="Training", initial=start_iter, total=self.config.max_iters)

        for step in pbar:
            lr = self.get_lr(step)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            # Gradient accumulation loop
            self.optimizer.zero_grad()
            accum_loss = 0.0
            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)
                x, y = x.to(self.config.device), y.to(self.config.device)
                
                with torch.autocast(device_type='cuda' if self.config.device == 'cuda' else 'cpu', dtype=torch.float16, enabled=(self.config.device == 'cuda')):
                    _, loss = self.model(x, y)
                
                self.scaler.scale(loss / self.config.gradient_accumulation_steps).backward()
                accum_loss += loss.item() / self.config.gradient_accumulation_steps

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            pbar.set_postfix({'loss': f'{accum_loss:.4f}', 'lr': f'{lr:.1e}'})

            if step % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"\nIter {step:5d}  train={losses['train']:.4f}  "
                      f"val={losses['val']:.4f}  lr={lr:.1e}")
                with open(self.log_path, 'a') as f:
                    f.write(f"{step},{losses['train']:.4f},{losses['val']:.4f},{lr:.1e}\n")
                if losses['val'] < self.best_val:
                    self.best_val = losses['val']
                    self.save_checkpoint(step, losses['val'])
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= 5:
                        print(f"Early stopping triggered at iter {step}")
                        break
