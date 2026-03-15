from config import Qwen3Config
from model import Qwen3
from safetensors.torch import load_file
from pathlib import Path


def detect_config(hf: dict) -> Qwen3Config:
    """
    Derive all architecture hyperparameters directly from weight shapes.
    Works for any Qwen3 model size without needing config.json.
    """
    cfg = Qwen3Config()

    # vocab_size, hidden_size
    emb = hf["model.embed_tokens.weight"]
    cfg.vocab_size, cfg.hidden_size = emb.shape

    # num_hidden_layers — count layer keys
    cfg.num_hidden_layers = (
        max(int(k.split(".")[2]) for k in hf if k.startswith("model.layers.")) + 1
    )

    # head_dim, num_attention_heads, num_key_value_heads
    # q_proj: [num_heads * head_dim, hidden]
    # k_proj: [num_kv_heads * head_dim, hidden]
    # q_norm: [head_dim]  ← most reliable source of head_dim
    qn = hf["model.layers.0.self_attn.q_norm.weight"]
    cfg.head_dim = qn.shape[0]  # 128 for all Qwen3 sizes

    qw = hf["model.layers.0.self_attn.q_proj.weight"]
    kw = hf["model.layers.0.self_attn.k_proj.weight"]
    cfg.num_attention_heads = qw.shape[0] // cfg.head_dim
    cfg.num_key_value_heads = kw.shape[0] // cfg.head_dim

    # intermediate_size
    cfg.intermediate_size = hf["model.layers.0.mlp.gate_proj.weight"].shape[0]

    # tie_word_embeddings
    cfg.tie_word_embeddings = "lm_head.weight" not in hf

    print(f"Detected config:")
    print(f"  vocab_size={cfg.vocab_size}, hidden={cfg.hidden_size}")
    print(
        f"  layers={cfg.num_hidden_layers}, heads={cfg.num_attention_heads}"
        f", kv_heads={cfg.num_key_value_heads}, head_dim={cfg.head_dim}"
    )
    print(
        f"  intermediate={cfg.intermediate_size}"
        f", tied_embeddings={cfg.tie_word_embeddings}"
    )

    return cfg


def remap_weights(hf: dict, cfg: Qwen3Config) -> dict:
    out = {}

    out["embed_tokens.weight"] = hf["model.embed_tokens.weight"]
    out["norm.weight"] = hf["model.norm.weight"]
    if not cfg.tie_word_embeddings:
        out["lm_head.weight"] = hf["lm_head.weight"]

    for i in range(cfg.num_hidden_layers):
        ph = f"model.layers.{i}"  # HF prefix
        pm = f"layers.{i}"  # our prefix

        out[f"{pm}.norm1.weight"] = hf[f"{ph}.input_layernorm.weight"]
        out[f"{pm}.norm2.weight"] = hf[f"{ph}.post_attention_layernorm.weight"]

        out[f"{pm}.attn.q_proj.weight"] = hf[f"{ph}.self_attn.q_proj.weight"]
        out[f"{pm}.attn.k_proj.weight"] = hf[f"{ph}.self_attn.k_proj.weight"]
        out[f"{pm}.attn.v_proj.weight"] = hf[f"{ph}.self_attn.v_proj.weight"]
        out[f"{pm}.attn.o_proj.weight"] = hf[f"{ph}.self_attn.o_proj.weight"]
        out[f"{pm}.attn.q_norm.weight"] = hf[f"{ph}.self_attn.q_norm.weight"]
        out[f"{pm}.attn.k_norm.weight"] = hf[f"{ph}.self_attn.k_norm.weight"]

        out[f"{pm}.ffn.gate.weight"] = hf[f"{ph}.mlp.gate_proj.weight"]
        out[f"{pm}.ffn.up.weight"] = hf[f"{ph}.mlp.up_proj.weight"]
        out[f"{pm}.ffn.down.weight"] = hf[f"{ph}.mlp.down_proj.weight"]

    return out


def load_model(
    model_path: str | None = None,
    model_id: str = "Qwen/Qwen3-0.6B",
    device: str = "cpu",
) -> tuple[Qwen3, str]:
    if model_path is not None:
        hf_state = load_file(model_path)
        local_dir = Path(model_path).parent
    else:
        from huggingface_hub import snapshot_download

        print(f"Downloading {model_id} ...")
        local_dir = snapshot_download(model_id, ignore_patterns=["*.msgpack", "*.h5"])

        # Load all safetensor shards
        shards = sorted(Path(local_dir).glob("*.safetensors"))
        hf_state = {}
        for s in shards:
            hf_state.update(load_file(str(s), device=device))
        print(f"Loaded {len(hf_state)} tensors from {len(shards)} shard(s)")

    # Auto-detect config from weight shapes — works for any Qwen3 size
    cfg = detect_config(hf_state)
    model = Qwen3(cfg)

    remapped = remap_weights(hf_state, cfg)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"[WARN] Missing    ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"[WARN] Unexpected ({len(unexpected)}): {unexpected[:8]}")
    if not missing and not unexpected:
        print("✓ All weights loaded cleanly.")

    return model.to(device).eval(), Path(local_dir)


def format_prompt(
    user_msg: str, system: str = "You are a helpful assistant.", reasoning=False
) -> str:
    if not reasoning:
        return (
            f"/no_think\n"
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"/no_think\n"
        )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
