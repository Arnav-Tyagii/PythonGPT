import torch

def generate_code(prompt, model, tokenizer, config,
                  max_new_tokens=200, temperature=0.8,
                  top_k=40, top_p=0.95, eos_token_id=None) -> str:
    model.eval()
    if eos_token_id is None:
        eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=config.device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens, temperature, top_k, top_p, eos_token_id=eos_token_id)
    full = tokenizer.decode(out[0].tolist())
    result = full[len(prompt):]   # strip the original prompt
    if result.endswith("<|endoftext|>"):
        result = result[:-len("<|endoftext|>")]
    return result

def generate_with_constraints(prompt, model, tokenizer, config,
                               max_new_tokens=500, temperature=0.8,
                               top_k=40, top_p=0.95,
                               use_constraints=True,
                               repetition_penalty=1.3) -> dict:
    from inference.constrained_decode import constrained_generate
    return constrained_generate(
        prompt, model, tokenizer, config,
        max_new_tokens, temperature, top_k, top_p,
        use_constraints, repetition_penalty
    )
