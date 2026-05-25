import os
import ast
import hashlib
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

def download_python_corpus(output_path: str, target_mb: int = 50):
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        
    ds = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        split="train",
        streaming=True
    )
    
    seen_hashes = set()
    files_checked = 0
    files_kept = 0
    collected_bytes = 0
    target_bytes = target_mb * 1024 * 1024
    
    pbar = tqdm(total=target_bytes, unit='B', unit_scale=True, desc="Collecting corpus")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in ds:
            files_checked += 1
            content = item['content']
            
            # Condition 1: size
            if not (500 <= len(content) <= 50000):
                continue
                
            # Condition 2: docstring
            if content.count('"""') < 2 and content.count("'''") < 2:
                continue
                
            # Condition 3: inline comments
            if content.count('#') < 3:
                continue
                
            # Condition 4: has function
            if 'def ' not in content:
                continue
                
            # Condition 5: valid python
            try:
                ast.parse(content)
            except SyntaxError:
                continue
                
            # Condition 6: predominantly ASCII
            ascii_ratio = len([c for c in content if ord(c) < 128]) / len(content)
            if ascii_ratio < 0.95:
                continue
                
            # Deduplicate
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            
            # Keep
            files_kept += 1
            text_to_write = content + '\n<|endoftext|>\n'
            byte_len = len(text_to_write.encode('utf-8'))
            f.write(text_to_write)
            
            collected_bytes += byte_len
            pbar.update(byte_len)
            
            if collected_bytes >= target_bytes:
                break
                
    pbar.close()
    print(f"\nCollection complete.")
    print(f"Files checked: {files_checked}")
    print(f"Files kept: {files_kept}")
    print(f"MB collected: {collected_bytes / (1024*1024):.2f}")
