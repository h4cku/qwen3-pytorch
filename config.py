from pydantic.dataclasses import dataclass


@dataclass
class Qwen3Config:
    # These will be overridden automatically from the checkpoint
    vocab_size: int = 151936
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    # RoPE
    rope_theta: float = 1_000_000.0
    max_seq_len: int = 4096
    # Norm
    rms_norm_eps: float = 1e-6
    # Misc
    tie_word_embeddings: bool = True
    eos_token_id: int = 151645  # <|im_end|>
