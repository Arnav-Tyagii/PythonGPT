import os
from data.download_stack import download_python_corpus
from data.preprocess import preprocess_corpus
from tokenizer.bpe_tokenizer import BPETokenizer

def main():
    raw_path = "data/python_raw.txt"
    clean_path = "data/python_corpus.txt"
    vocab_dir = "tokenizer/vocab"
    
    os.makedirs("data", exist_ok=True)
    
    print("Step 1: Downloading and filtering Python corpus...")
    download_python_corpus(raw_path, target_mb=1000)
    
    print("\nStep 2: Preprocessing and cleaning...")
    preprocess_corpus(raw_path, clean_path)
    
    if os.path.exists(clean_path):
        file_size_mb = os.path.getsize(clean_path) / (1024 * 1024)
        print(f"\nCorpus stats: Size={file_size_mb:.2f} MB")
    
    print("\nStep 3: Training BPE Tokenizer...")
    tok = BPETokenizer(vocab_size=8000)
    tok.train(clean_path, vocab_dir)
    
    print("\nDone! You can now run `python train.py`")

if __name__ == "__main__":
    main()
