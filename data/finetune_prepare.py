import os
import random

def prepare_finetune_data(algo_path: str, original_corpus_path: str, output_path: str, algo_repeat: int = 20):
    print("Reading algorithm dataset...")
    with open(algo_path, "r", encoding="utf-8") as f:
        algo_content = f.read()
    
    # Split by the document separator to interleave chunks properly
    algo_docs = algo_content.split("\n\n<|endoftext|>\n\n")
    algo_docs = [doc for doc in algo_docs if doc.strip()]
    
    print(f"Found {len(algo_docs)} algorithm documents. Repeating {algo_repeat} times...")
    repeated_algo_docs = algo_docs * algo_repeat
    
    print("Reading original corpus sample (50MB)...")
    # Read up to ~50MB from original corpus
    # If the file is large, we can read chunks until we hit 50MB.
    # The original file is separated by \n\n<|endoftext|>\n\n
    original_docs = []
    current_size = 0
    target_size = 50 * 1024 * 1024 # 50 MB
    
    with open(original_corpus_path, "r", encoding="utf-8") as f:
        # Since reading entire 1GB into memory might be slow but feasible, 
        # let's read the whole thing and sample if memory allows, or read line by line.
        # For simplicity, we read everything and split, then take a random sample.
        # But to be safer with memory, we chunk read.
        pass
        
    # Better approach for random 50MB sample without full 1GB memory load:
    # Actually, we can read the whole thing if it's 1GB, Python handles 1GB strings fine,
    # but let's be efficient. We will read chunks and accumulate.
    with open(original_corpus_path, "r", encoding="utf-8") as f:
        while current_size < target_size:
            # Read a 1MB chunk
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            original_docs.append(chunk)
            current_size += len(chunk.encode('utf-8'))
            
    # Combine original_docs string and split to get proper document boundaries
    original_text = "".join(original_docs)
    original_chunks = original_text.split("\n\n<|endoftext|>\n\n")
    original_chunks = [doc for doc in original_chunks if doc.strip()]
    
    # Mix: 1 chunk of original, 2 chunks of algorithms
    # A generic interleave based on ratios, but let's just combine and shuffle
    # The user asked: "Interleave: for every 1 chunk of original, add 2 chunks of algorithms. Shuffle the combined dataset."
    # We will build a list applying the 2:1 ratio and then shuffle it.
    
    combined_docs = []
    algo_idx = 0
    orig_idx = 0
    
    while algo_idx < len(repeated_algo_docs) or orig_idx < len(original_chunks):
        # 1 original chunk
        if orig_idx < len(original_chunks):
            combined_docs.append(original_chunks[orig_idx])
            orig_idx += 1
            
        # 2 algo chunks
        for _ in range(2):
            if algo_idx < len(repeated_algo_docs):
                combined_docs.append(repeated_algo_docs[algo_idx])
                algo_idx += 1

    print("Shuffling combined dataset...")
    random.shuffle(combined_docs)
    
    # Write to output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n<|endoftext|>\n\n".join(combined_docs))
        
    algo_tokens = sum(len(doc.split()) for doc in repeated_algo_docs)
    orig_tokens = sum(len(doc.split()) for doc in original_chunks)
    total_tokens = algo_tokens + orig_tokens
    
    print(f"\nDataset Preparation Complete!")
    print(f"Algorithm tokens (approx): {algo_tokens:,}")
    print(f"Original tokens (approx): {orig_tokens:,}")
    print(f"Total tokens (approx): {total_tokens:,}")
    print(f"Mix ratio (Algo:Orig): {algo_tokens / max(1, orig_tokens):.2f}")
    print(f"Saved mixed dataset to {output_path}")

if __name__ == "__main__":
    prepare_finetune_data(
        algo_path="data/algorithms_corpus.txt",
        original_corpus_path="data/python_corpus.txt",
        output_path="data/finetune_corpus.txt",
        algo_repeat=20
    )
