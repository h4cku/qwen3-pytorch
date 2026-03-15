import torch
import torch.nn as nn
from config import Qwen3Config
from typing import Optional
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # ones (different from Qwen3.5!)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.float()
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(dtype)  # weight * normed (not 1+weight)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Matches Qwen3RotaryEmbedding:
      inv_freq over arange(0, head_dim, 2)  → length head_dim//2
      freqs = outer(positions, inv_freq)    → [T, head_dim//2]
      emb   = cat(freqs, freqs)             → [T, head_dim]
      returns cos(emb), sin(emb)            each [T, head_dim]
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
    return emb.cos(), emb.sin()  # each [T, head_dim]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset=0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k   : [B, H, T, head_dim]
    cos/sin: [T, head_dim]   (full head_dim, already doubled)
    Matches apply_rotary_pos_emb with unsqueeze_dim=1.
    """
    T = q.shape[2]
    c = cos[offset : offset + T].unsqueeze(0).unsqueeze(0)
    s = sin[offset : offset + T].unsqueeze(0).unsqueeze(0)
    q_out = q * c + rotate_half(q) * s
    k_out = k * c + rotate_half(k) * s
    return q_out, k_out


class Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        nq = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads
        D = cfg.head_dim
        hid = cfg.hidden_size

        self.nq, self.nkv, self.D = nq, nkv, D
        self.num_kv_groups = nq // nkv
        self.scale = D**-0.5

        # No *2 on q_proj — Qwen3 has no output gate
        self.q_proj = nn.Linear(hid, nq * D, bias=False)
        self.k_proj = nn.Linear(hid, nkv * D, bias=False)
        self.v_proj = nn.Linear(hid, nkv * D, bias=False)
        self.o_proj = nn.Linear(nq * D, hid, bias=False)

        # Per-head QK norms (same as Qwen3.5 but using ones-init RMSNorm)
        self.q_norm = RMSNorm(D, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(D, cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, tuple]:
        B, T, _ = x.shape
        nq, nkv, D = self.nq, self.nkv, self.D

        # Project, reshape, apply QK-norm
        # Matches: q_norm(q_proj(x).view(...)).transpose(1,2)
        q = self.q_norm(self.q_proj(x).view(B, T, nq, D)).transpose(1, 2)  # [B,nq,T,D]
        k = self.k_norm(self.k_proj(x).view(B, T, nkv, D)).transpose(
            1, 2
        )  # [B,nkv,T,D]
        v = self.v_proj(x).view(B, T, nkv, D).transpose(1, 2)  # [B,nkv,T,D]

        # RoPE (full head_dim)
        offset = kv_cache[0].shape[2] if kv_cache is not None else 0
        q, k = apply_rope(q, k, cos, sin, offset=offset)

        # KV cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)
        S = k.shape[2]

        # GQA expand KV
        k = k.repeat_interleave(self.num_kv_groups, dim=1)  # [B,nq,S,D]
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,nq,T,S]
        if T > 1:
            mask = torch.full((T, S), float("-inf"), device=x.device, dtype=x.dtype)
            mask = torch.zeros((T, S), device=x.device, dtype=torch.bool)
            mask[:, : S - T] = False  # past tokens always visible
            causal_part = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            mask[:, S - T :] = causal_part
            attn = attn.masked_fill(mask[None, None], float("-inf"))
        attn = F.softmax(attn, dim=-1)

        # No output gate here — plain attention output
        out = (attn @ v).transpose(1, 2).reshape(B, T, nq * D)  # [B,T,nq*D]
        return self.o_proj(out), new_cache


class FeedForward(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Qwen3Block(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.attn = Attention(cfg)
        self.ffn = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)  # input_layernorm
        self.norm2 = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps
        )  # post_attention_layernorm

    def forward(self, x: torch.Tensor, cos, sin, cache=None):
        # Pre-norm → attention → residual
        h, new_cache = self.attn(self.norm1(x), cos, sin, cache)
        x = x + h
        # Pre-norm → FFN → residual
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class Qwen3(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3Block(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Build RoPE cache (CPU, moved to device on first forward)
        cos, sin = build_rope_cache(
            cfg.max_seq_len, cfg.head_dim, cfg.rope_theta, torch.device("cpu")
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self, input_ids: torch.Tensor, past_caches: Optional[list] = None
    ) -> tuple[torch.Tensor, list]:
        device = input_ids.device
        x = self.embed_tokens(input_ids)
        cos = self.rope_cos.to(device)
        sin = self.rope_sin.to(device)

        caches = []
        for i, layer in enumerate(self.layers):
            x, c = layer(x, cos, sin, past_caches[i] if past_caches else None)
            caches.append(c)

        return self.lm_head(self.norm(x)), caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        logits, caches = self(input_ids)
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            nl = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-p nucleus sampling
            sl, si = torch.sort(nl, descending=True)
            p = F.softmax(sl, dim=-1)
            cp = p.cumsum(-1)
            sl[(cp - p) > top_p] = float("-inf")
            p = F.softmax(sl, dim=-1)
            nt = si.gather(-1, torch.multinomial(p, 1))  # [B, 1]

            generated = torch.cat([generated, nt], dim=1)
            if nt.item() == self.cfg.eos_token_id:
                break

            logits, caches = self(nt, caches)

        return generated
