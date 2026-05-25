class PythonConstraintChecker:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Precompute token id → string lookup once for speed
        self.id_to_str = {}
        for i in range(tokenizer.vocab_size):
            try:
                self.id_to_str[i] = tokenizer.decode([i])
            except:
                self.id_to_str[i] = ''

    def is_recoverable(self, code: str) -> bool:
        # Wrap EVERYTHING in try/except — never raise, always return bool
        try:
            # Rule 1: bracket balance
            open_count  = code.count('(') + code.count('[') + code.count('{')
            close_count = code.count(')') + code.count(']') + code.count('}')
            if close_count > open_count:
                return False

            # Rule 2: indentation check via tokenize
            import tokenize, io
            try:
                list(tokenize.generate_tokens(io.StringIO(code).readline))
            except tokenize.TokenError:
                pass   # incomplete code — fine
            except IndentationError:
                return False  # unrecoverable

            return True
        except:
            return True  # fail open — never block on unexpected error

    def get_valid_token_mask(self,
                              current_code: str,
                              candidate_token_ids: list) -> set:
        valid = set()
        for tid in candidate_token_ids:
            tok_str = self.id_to_str.get(tid, '')
            try:
                if self.is_recoverable(current_code + tok_str):
                    valid.add(tid)
            except:
                valid.add(tid)  # fail open

        # Safety fallback: never return empty set
        if not valid:
            return set(candidate_token_ids)
        return valid

def fix_common_errors(code: str) -> str:
    import ast
    # 1. Try if valid already
    try:
        ast.parse(code)
        return code
    except:
        pass
        
    # 2. Try replacing ''' with """
    fixed_quotes = code.replace("'''", '"""')
    try:
        ast.parse(fixed_quotes)
        return fixed_quotes
    except:
        pass
        
    # 3. If last non-empty line ends with ':', append '\n    pass'
    lines = code.splitlines()
    if lines:
        for i in range(len(lines)-1, -1, -1):
            if lines[i].strip():
                if lines[i].rstrip().endswith(':'):
                    fixed_pass = code + '\n    pass\n'
                    try:
                        ast.parse(fixed_pass)
                        return fixed_pass
                    except:
                        pass
                break
                
    # 4. Return original if nothing worked
    return code



def constrained_generate(
    prompt: str,
    model,
    tokenizer,
    config,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    use_constraints: bool = True,
    repetition_penalty: float = 1.3,
) -> dict:
    import torch
    import torch.nn.functional as F

    model.eval()

    # Get EOS token id
    try:
        eos_id = tokenizer.encode('<|endoftext|>')[0]
    except:
        eos_id = None

    # Encode prompt
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded], dtype=torch.long, device=config.device)

    checker = PythonConstraintChecker(tokenizer) if use_constraints else None
    constraints_applied = 0
    fallbacks = 0
    stopped_naturally = False
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]              # (1, vocab_size)

            # Temperature
            logits = logits / temperature

            # Repetition penalty on recent tokens
            if repetition_penalty != 1.0:
                recent = idx[0, -64:].tolist()
                for tid in set(recent):
                    if 0 <= tid < logits.size(-1):
                        if logits[0, tid] > 0:
                            logits[0, tid] /= repetition_penalty
                        else:
                            logits[0, tid] *= repetition_penalty

            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p nucleus sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(
                    1, sorted_idx, sorted_logits)

            # Constrained decoding
            if use_constraints and checker is not None:
                # Get surviving candidate ids
                surviving = (logits[0] > float('-inf')).nonzero(
                    as_tuple=True)[0].tolist()

                if surviving:
                    # Decode current full output
                    current_ids = idx[0].tolist() + generated_ids
                    try:
                        current_code = tokenizer.decode(
                            idx[0].tolist()) + tokenizer.decode(generated_ids)
                    except:
                        current_code = prompt

                    valid_set = checker.get_valid_token_mask(
                        current_code, surviving)

                    blocked = 0
                    for tid in surviving:
                        if tid not in valid_set:
                            logits[0, tid] = float('-inf')
                            blocked += 1

                    constraints_applied += blocked
                    if blocked == len(surviving):
                        fallbacks += 1

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (1, 1)
            next_id = next_tok.item()

            # EOS check
            if eos_id is not None and next_id == eos_id:
                stopped_naturally = True
                break

            generated_ids.append(next_id)
            idx = torch.cat([idx, next_tok], dim=1)

    # Decode output, strip prompt
    try:
        raw_full_text = tokenizer.decode(idx[0].tolist())
        fixed_full = fix_common_errors(raw_full_text)
        was_fixed = fixed_full != raw_full_text
        generated = fixed_full[len(prompt):]
    except:
        fixed_full = prompt
        generated = ''
        was_fixed = False

    return {
        'code': generated,
        'full': fixed_full,
        'stopped_naturally': stopped_naturally,
        'constraints_applied': constraints_applied,
        'fallbacks': fallbacks,
        'was_fixed': was_fixed,
    }
