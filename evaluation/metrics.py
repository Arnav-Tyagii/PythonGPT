import math
import ast
import torch
from inference.generate import generate_code

def compute_perplexity(model, val_loader, config) -> float:
    losses = []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= 50: break
            x, y = x.to(config.device), y.to(config.device)
            _, loss = model(x, y)
            losses.append(loss.item())
    if not losses:
        return 0.0
    return math.exp(sum(losses) / len(losses))

def check_syntax(code: str) -> dict:
    try:
        ast.parse(code)
        return {'valid': True, 'error': None}
    except SyntaxError as e:
        return {'valid': False, 'error': str(e)}

def evaluate_samples(model, tokenizer, config, n=5) -> dict:
    prompts = [
        'def calculate_mean(numbers):\n    """',
        'def binary_search(arr, target):\n    """',
        'class DataProcessor:\n    """',
        'def read_csv_file(filepath):\n    # Read CSV and return',
        'import numpy as np\n\ndef normalize_array(arr):',
    ]
    results = {'valid': 0, 'total': 0, 'samples': []}
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    for p in prompts[:n]:
        gen = generate_code(p, model, tokenizer, config, max_new_tokens=500, eos_token_id=eos_token_id)
        
        # Truncation strategy
        truncated = gen
        idx_def = truncated.find('\ndef ')
        idx_cls = truncated.find('\nclass ')
        
        idxs = [i for i in (idx_def, idx_cls) if i != -1]
        if idxs:
            truncated = truncated[:min(idxs)]
        else:
            # If no new function/class is started, drop the potentially cut-off last line
            if not gen.endswith('\n'):
                last_nl = truncated.rfind('\n')
                if last_nl != -1:
                    truncated = truncated[:last_nl+1]

        check = check_syntax(p + truncated)
        results['total'] += 1
        if check['valid']: 
            results['valid'] += 1
        results['samples'].append({
            'prompt': p, 
            'output': truncated,
            'valid': check['valid']
        })
    results['syntax_rate'] = results['valid'] / results['total'] if results['total'] > 0 else 0.0
    return results
