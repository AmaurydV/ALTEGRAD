from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_gpt2(device):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
    model.eval()
    return tokenizer, model


def build_prompt(test_sig, retrieved_descs):
    """
    test_sig: (num_atoms, num_bonds, density)
    retrieved_descs: list[str]
    """
    n, m, d = test_sig

    prompt = (
        "You are a chemistry expert.\n"
        "Below are descriptions of molecules that are structurally similar.\n"
        "Based on them, write a concise and accurate description of the target molecule.\n\n"
        f"Target molecule properties:\n"
        f"- Number of atoms: {n}\n"
        f"- Number of bonds: {m}\n"
        f"- Bond density: {d:.2f}\n\n"
        "Similar molecule descriptions:\n"
    )

    for i, desc in enumerate(retrieved_descs):
        prompt += f"{i+1}. {desc.strip()}\n"

    prompt += "\nSynthesized description:"
    return prompt


@torch.no_grad()
def gpt2_synthesize(prompt, tokenizer, model, device, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Keep only generated part
    return text.split("Synthesized description:")[-1].strip()
