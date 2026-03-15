# Qwen3-PyTorch

This repository provides a clean, minimalistic implementation of the Qwen3 large language model in PyTorch. It focuses on clarity and direct implementation of the core architecture, including Grouped Query Attention (GQA) and Rotary Positional Embeddings (RoPE).

## How to Run

To run the main program and generate text with the Qwen3 model, follow these steps:

1.  **Ensure Model Weights:** The `main.py` script expects model weights to be present in `./hf/model.safetensors` or will attempt to download them from Hugging Face if a `model_id` is provided. You can place your `model.safetensors` and `tokenizer.json` files in the `hf/` directory.

2.  **Execute the Script:**
    You can run the script with the default Qwen3-0.6B model:

    ```bash
    python main.py
    ```

    Or, specify a different Qwen3 model ID as a command-line argument:

    ```bash
    python main.py "Qwen/Qwen3-1.8B"
    ```

    The script will load the model, encode a sample prompt, and generate a response.

## Model Architecture (Qwen3)

The Qwen3 model implemented here is built upon a decoder-only transformer architecture. Key components and their interactions are detailed below:

### Core Components

*   **`Qwen3`**: The main model class, orchestrating the entire forward pass. It comprises an embedding layer, a stack of `Qwen3Block`s, a final normalization, and a language modeling head.
*   **`Qwen3Block`**: Represents a single transformer block. Each block includes an Attention mechanism and a Feed-Forward Network, both preceded by RMSNorm layers, and incorporates residual connections.
*   **`Attention`**: Implements the multi-head (or grouped-query) attention mechanism. It uses linear projections for queries, keys, and values, applies Rotary Positional Embeddings (RoPE), and utilizes Grouped Query Attention (GQA) for efficiency. QK normalization is applied per-head.
*   **`FeedForward`**: A standard feed-forward network with three linear layers (`gate`, `up`, `down`) and a SiLU activation function, forming a SwiGLU-like structure.
*   **`RMSNorm`**: A root mean square normalization layer used for stabilizing training, applied before attention and feed-forward networks within each block, and once at the model's output.
*   **Rotary Positional Embeddings (RoPE)**: Applied within the `Attention` mechanism to inject positional information into the query and key vectors.

### Architectural Flow

The model processes input tokens through the following sequence:

1.  **Token Embedding**: Input token IDs are converted into dense vector representations.
2.  **Stacked Decoder Blocks**: The embeddings pass through multiple `Qwen3Block`s. Each block performs:
    *   **Pre-Normalization**: An `RMSNorm` layer.
    *   **Attention**: Computes self-attention over the sequence, incorporating RoPE and GQA.
    *   **Residual Connection**: The attention output is added to the input of the block.
    *   **Pre-Normalization**: Another `RMSNorm` layer.
    *   **Feed-Forward Network**: Processes the features with a non-linear transformation.
    *   **Residual Connection**: The FFN output is added back.
3.  **Final Normalization**: An `RMSNorm` layer is applied to the output of the last decoder block.
4.  **Language Modeling Head**: A linear layer projects the normalized features to the vocabulary size, producing logits for the next token prediction.

## Model Architecture Diagram

```mermaid
graph TD
    A[Input IDs] --> B(Embed Tokens)
    B --> C(RoPE Cache)
    C --> BLK

    subgraph BLK["Qwen3Block × N layers"]
        QB_In[Input] --> QB_Norm1(RMSNorm)
        QB_Norm1 --> ATTN

        subgraph ATTN["Attention"]
            Attn_In[Input] --> Attn_Q(q_proj)
            Attn_In --> Attn_K(k_proj)
            Attn_In --> Attn_V(v_proj)
            Attn_Q --> Attn_QN(q_norm)
            Attn_K --> Attn_KN(k_norm)
            Attn_QN --> Attn_RoPE(Apply RoPE)
            Attn_KN --> Attn_RoPE
            Attn_RoPE --> Attn_Cache(KV Cache)
            Attn_V --> Attn_Cache
            Attn_Cache --> Attn_GQA(GQA repeat_interleave)
            Attn_GQA --> Attn_SDP(Scaled dot-product + causal mask)
            Attn_SDP --> Attn_O(o_proj)
        end

        Attn_O --> QB_Res1([+ residual])
        QB_In --> QB_Res1
        QB_Res1 --> QB_Norm2(RMSNorm)
        QB_Norm2 --> FFN

        subgraph FFN["FeedForward"]
            FFN_In[Input] --> FFN_Gate(gate_proj)
            FFN_In --> FFN_Up(up_proj)
            FFN_Gate --> FFN_SiLU(SiLU)
            FFN_SiLU --> FFN_Mul(× element-wise)
            FFN_Up --> FFN_Mul
            FFN_Mul --> FFN_Down(down_proj)
        end

        FFN_Down --> QB_Res2([+ residual])
        QB_Res1 --> QB_Res2
        QB_Res2 --> QB_Out[Output]
    end

    QB_Out --> D(RMSNorm)
    D --> E(lm_head)
    E --> F[Output logits]
```
