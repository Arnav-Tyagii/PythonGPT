import urllib.request
import json
import ast
import time
import os
from tqdm import tqdm

def download_algorithms_dataset(output_path: str):
    repos = [
        {
            "tree_url": "https://api.github.com/repos/TheAlgorithms/Python/git/trees/master?recursive=1",
            "raw_base": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/"
        },
        {
            "tree_url": "https://api.github.com/repos/keon/algorithms/git/trees/master?recursive=1",
            "raw_base": "https://raw.githubusercontent.com/keon/algorithms/master/"
        },
        {
            "tree_url": "https://api.github.com/repos/OmkarPathak/pygorithm/git/trees/master?recursive=1",
            "raw_base": "https://raw.githubusercontent.com/OmkarPathak/pygorithm/master/"
        }
    ]

    all_py_files = []
    
    print("Fetching repository trees...")
    for repo in repos:
        try:
            req = urllib.request.Request(repo["tree_url"], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
                for item in data.get("tree", []):
                    if item.get("type") == "blob" and item.get("path", "").endswith(".py"):
                        all_py_files.append((repo["raw_base"] + item["path"]))
        except Exception as e:
            print(f"Error fetching tree from {repo['tree_url']}: {e}")

    print(f"Found {len(all_py_files)} total .py files. Downloading and filtering...")
    
    kept_files = 0
    total_size_bytes = 0
    accepted_contents = []

    for url in tqdm(all_py_files, desc="Downloading files"):
        time.sleep(0.1) # Rate limiting
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8', errors='ignore')
                
                # Filters
                if len(content) < 200:
                    continue
                if 'def ' not in content:
                    continue
                if content.count('#') < 2:
                    continue
                
                try:
                    ast.parse(content)
                except SyntaxError:
                    continue
                
                accepted_contents.append(content)
                kept_files += 1
                total_size_bytes += len(content.encode('utf-8'))
        except Exception:
            # Handle 404s and network errors gracefully
            continue

    print(f"\nFinished processing. Kept {kept_files} files.")
    print(f"Total size: {total_size_bytes / 1024:.2f} KB")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n<|endoftext|>\n\n".join(accepted_contents))
    
    print(f"Saved algorithm dataset to {output_path}")

if __name__ == "__main__":
    download_algorithms_dataset("data/algorithms_corpus.txt")
