def score_sequences(model, tokenizer, chat_prompts, completions, max_length=1024):
    tokenizer.padding_side = 'right'
    full_texts = [p + c for p, c in zip(chat_prompts, completions)]
    prompt_lens = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in chat_prompts]

    enc = tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True,
                     max_length=max_length).to(model.device)
    out = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])

    logits = out.logits[:, :-1, :]            # keep model dtype (bf16) -- do NOT .float() the whole tensor
    target_ids = enc['input_ids'][:, 1:]

    # FIX: cross_entropy's fused kernel computes log_softmax + NLL internally
    # without ever materializing a separate [batch, seq_len, vocab] float32
    # tensor the way log_softmax(...).gather(...) does. Same math, far less
    # peak memory -- this is what was OOMing on a 152k-token vocab.
    token_nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        reduction='none',
    ).view(target_ids.shape)
    token_logprobs = -token_nll                # [B, L-1]

    attn = enc['attention_mask'][:, 1:]
    seq_len = attn.shape[1]
    completion_mask = torch.zeros_like(attn)
    for row, plen in enumerate(prompt_lens):
        start = max(min(plen - 1, seq_len), 0)
        completion_mask[row, start:] = attn[row, start:]

    del out, logits    # release the big logits tensor before returning, not at next GC cycle
    return token_logprobs, completion_mask.float()
