import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TextDataset(Dataset):
    def __init__(self, token_ids, block_size: int):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.token_ids[idx:idx+self.block_size].astype(np.int64), dtype=torch.long)
        y = torch.tensor(self.token_ids[idx+1:idx+self.block_size+1].astype(np.int64), dtype=torch.long)
        return x, y

import os
from tqdm import tqdm

def get_dataloaders(config, tokenizer):
    cache_path = config.data_path + ".npy"
    if os.path.exists(cache_path):
        print(f"Loading cached tokenized corpus from {cache_path}...")
        token_ids = np.load(cache_path)
        print(f"Token count: {len(token_ids):,}")
    else:
        print("Encoding corpus in chunks to save RAM...")
        chunk_size = 1024 * 512  # 500 KB chunks (much faster for the BPE merge graph)
        token_ids_list = []
        
        file_size = os.path.getsize(config.data_path)
        
        with open(config.data_path, 'r', encoding='utf-8') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Encoding") as pbar:
                while True:
                    lines = f.readlines(chunk_size)
                    if not lines:
                        break
                    chunk = "".join(lines)
                    ids = tokenizer.encode(chunk)
                    # Store as uint16 to save massive amounts of RAM (vocab is 8000 < 65535)
                    token_ids_list.append(np.array(ids, dtype=np.uint16))
                    pbar.update(len(chunk.encode('utf-8')))
                
        print("Concatenating chunks...")
        token_ids = np.concatenate(token_ids_list)
        print(f"Token count: {len(token_ids):,}")
        
        print("Saving tokenized corpus to cache...")
        np.save(cache_path, token_ids)
    
    split_idx = int(len(token_ids) * 0.9)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    
    train_ds = TextDataset(train_ids, config.block_size)
    val_ds = TextDataset(val_ids, config.block_size)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=(config.device == 'cuda')
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=(config.device == 'cuda')
    )
    
    return train_loader, val_loader
