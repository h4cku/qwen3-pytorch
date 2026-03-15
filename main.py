import torch
from utils import load_model, format_prompt
from tokenizers import Tokenizer

if __name__ == "__main__":
    import sys

    MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}  |  Model: {MODEL_ID}")

    model, local_dir = load_model(model_path="./hf/model.safetensors", device=DEVICE)
    enc = Tokenizer.from_file("./hf/tokenizer.json")

    prompt = format_prompt(
        "What is the difference between a list and a tuple in Python?"
    )
    token_ids = enc.encode(prompt).ids
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    print(f"Prompt tokens: {len(token_ids)}")

    out = model.generate(input_ids, max_new_tokens=256, temperature=0.7, top_p=0.9)
    response = enc.decode(out[0, len(token_ids) :].tolist())
    print("\n" + "=" * 60 + "\nRESPONSE:\n" + "=" * 60)
    print(response)
