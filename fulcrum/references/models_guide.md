# Open-Weight Models Reference Guide

This guide provides a comprehensive taxonomy of leading open-weight LLMs, their architectural properties, memory requirements, and licensing considerations for deployment.

---

## Open Weights vs Proprietary API-Based Access

The `/fulcrum` skill is strictly dedicated to **open-weight, downloadable, and self-deployable models**. Understanding the technical distinction between downloadable open weights and proprietary API access is fundamental:

| Attribute | Downloadable Open Weights (Supported by `/fulcrum`) | Proprietary API-Based Access (Excluded) |
| :--- | :--- | :--- |
| **Model Weight Access** | Full access to tensor checkpoints (`.safetensors`, GGUF, AWQ, FP8). | Weights are completely hidden behind a cloud black-box. |
| **Hosting & Control** | Self-hosted on user-controlled infrastructure (e.g. GCP GCE with vLLM). | Hosted on third-party multi-tenant proprietary cloud servers. |
| **Data Privacy & Security**| Zero data egress; prompts never leave your private VPC or network. | Prompts and outputs transit external third-party APIs. |
| **Customisation & Tuning** | Custom inference flags, chunked prefill, speculative decoding, LoRA fine-tuning. | Limited to provider-exposed API parameters and system prompts. |
| **Cost Model** | Predictable hourly GPU compute pricing (e.g. preemptible/spot L4 instances). | Metered per-token API pricing that scales with query volume. |
| **Examples** | Google Gemma, Meta Llama, Alibaba Qwen, DeepSeek R1, Mistral. | OpenAI GPT-4o, Anthropic Claude 3.5, Google Gemini API endpoints. |

---

## Model Families & Benchmarks

### 1. Google Gemma 2 & Gemma 3
- **Architectures**: Decoder-only transformers incorporating sliding-window local-global attention and logit capping.
- **Variants**:
  - `google/gemma-3-1b-it`: Ultra-lightweight (1B), 32k context, ideal for edge, mobile, and speculative draft decoding.
  - `google/gemma-3-4b-it`: Highly capable 4B model, fits in <10 GB VRAM.
  - `google/gemma-3-12b-it`: Sweet-spot model delivering strong reasoning and code capability, fits easily on a single 24GB NVIDIA L4.
  - `google/gemma-3-27b-it`: Near-70B capability, fits on 2x L4 in BF16 or single L4 in FP8.
  - `google/gemma-2-9b-it`: Battle-tested 9B dense model with 8k context.
- **Licensing**: Gemma Terms of Use (permissive open-weights, free for research and commercial use subject to safety terms).
- **Gated**: Yes, requires accepting terms on Hugging Face and providing `HF_TOKEN`.

### 2. Meta Llama 3.1 & Llama 3.3
- **Architectures**: Standard dense decoder-only transformer with Grouped Query Attention (GQA) and RoPE context scaling up to 128k tokens.
- **Variants**:
  - `meta-llama/Llama-3.2-1B-Instruct` & `3B-Instruct`: Compact models for on-device and high-throughput summarisation.
  - `meta-llama/Llama-3.1-8B-Instruct`: Industry baseline for general tasks, tool use, and agentic workflows.
  - `meta-llama/Llama-3.3-70B-Instruct`: Flagship 70B model matching Llama 3.1 405B quality on many evaluations.
- **Licensing**: Llama 3.1 / 3.3 Community License (commercial use permitted up to 700M monthly active users).
- **Gated**: Yes, requires Hugging Face access request.

### 3. Alibaba Qwen 2.5 & Qwen 2.5 Coder
- **Architectures**: Dense transformer with GQA, SwiGLU activations, and wide multilingual tokeniser vocabulary (152k tokens).
- **Variants**:
  - `Qwen/Qwen2.5-Coder-7B-Instruct`: Exceptional code completion and agentic programming at 7B scale.
  - `Qwen/Qwen2.5-Coder-32B-Instruct`: Commercial-grade code generation rivalling top proprietary models.
  - `Qwen/Qwen2.5-7B-Instruct` & `32B-Instruct`: General-purpose instruction models with outstanding math and JSON extraction.
  - `Qwen/Qwen2.5-72B-Instruct`: Dense 72B powerhouse.
- **Licensing**: Apache-2.0 for 0.5B, 1.5B, 7B, 14B, 32B models. Permissive and ungated.

### 4. Alibaba Qwen 3 & Qwen 3.8
- **Architectures**: Hybrid architectures featuring alternating Gated DeltaNet (linear attention) and periodic full-attention layers with Grouped Query Attention (GQA), native vision-language multimodal capabilities, Multi-Token Prediction (MTP), and dynamic reasoning/thinking depth controls.
- **Variants**:
  - `Qwen/Qwen3-8B`: Dense 8B model with 128k context, excellent balance of tool-calling and reasoning. Fits in ~23 GB in native BF16 or ~11 GB in FP8.
  - `Qwen/Qwen3.8-27B`: Flagship 27B dense multimodal model with native 256k context. Featuring Gated DeltaNet attention (only 16 of 64 layers store KV cache). Fits in ~21.3 GB in 4-bit (AWQ/GPTQ/GGUF) or ~29 GB in FP8, delivering top-tier agentic and reasoning capabilities within 32 GB RAM.
  - `Qwen/Qwen3.8-2.4T-A95B`: 2.4T MoE powerhouse with 95B active parameters.
- **Licensing**: Apache-2.0. Ungated.

### 5. DeepSeek R1 & Distilled Models
- **Architectures**: Chain-of-thought reasoning models trained with large-scale reinforcement learning.
- **Variants**:
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` / `14B` / `32B`: Qwen 2.5 models fine-tuned on DeepSeek-R1 reasoning trajectories.
  - `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`: Llama 3.3 70B fine-tuned on R1 trajectories.
  - `deepseek-ai/DeepSeek-R1`: Massive 671B Mixture-of-Experts (37B active parameters per token). Requires cluster-level GPU deployment (e.g. 8x H100 in FP8).
- **Licensing**: MIT License (fully permissive, commercial use and modification unencumbered).

### 5. Mistral AI
- **Variants**:
  - `mistralai/Mistral-Nemo-Instruct-2407`: 12B model with 128k context, co-developed with NVIDIA.
  - `mistralai/Mistral-Small-24B-Instruct-2501`: 24B dense reasoning model.
- **Licensing**: Apache-2.0.

---

## Memory Footprint Calculations

When calculating GPU memory allocation for LLM serving, account for three distinct components:

### 1. Model Weights
$$\text{VRAM}_{\text{weights}} = \text{Parameters} \times \text{BytesPerParameter} \times 1.05$$

- **BF16 / FP16**: 2.0 bytes/parameter (e.g. 8B parameters = 16 GB).
- **FP8**: 1.0 byte/parameter (e.g. 8B parameters = 8 GB).
- **AWQ / GPTQ (4-bit)**: ~0.55 bytes/parameter (e.g. 8B parameters = 4.4 GB).

### 2. KV Cache
$$\text{VRAM}_{\text{KV}} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times \text{BytesPerElem} \times C_{\text{context}} \times B_{\text{batch}}$$

- For models with Grouped Query Attention (GQA), $N_{\text{kv\_heads}}$ is typically 4 or 8, dramatically reducing memory compared to full Multi-Head Attention.
- Setting `--gpu-memory-utilization 0.95` allows vLLM's PagedAttention allocator to dedicate all non-weight memory to KV cache blocks.

### 3. Activation & Workspace Headroom
- Typically requires 2 GB to 4 GB for CUDA/XLA memory pools, PyTorch activations, and kernel workspace.

---

## Model Architecture TPU Compatibility Tiers

Cloud TPUs run models through PyTorch/XLA and Pallas kernels. Understanding model architecture compatibility helps select the correct TPU slice:

| Tier | Compatibility Level | Model Families | TPU Execution Characteristics | Recommended Hardware |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Native / Flawless** | Qwen 3.8 / 2.5, Llama 3.1 / 3.3, Gemma 2 / 3, DeepSeek R1 Distill | Standard dense transformer; GQA shards cleanly across ICI mesh; maps directly to XLA matrix units (MXU). | `ct5lp-hightpu-1t/4t/8t`<br>`ct6e-standard-1t/4t/8t` |
| **Tier 1 / 2** | **Remote Code / PyTorch Ops** | GLM 5.3 / GLM-4 | Requires `--trust-remote-code`. Custom ops must execute via pure PyTorch/XLA rather than proprietary CUDA kernels. | `ct5lp-hightpu-4t/8t`<br>`ct6e-standard-4t` |
| **Tier 2** | **Memory Bandwidth & Long Context** | Kimi K3 (200k–1M context) | Extreme context requires Pallas PagedAttention (`VLLM_TPU_ENABLE_PALLAS=1`), chunked prefill, and prefix caching to prevent HBM exhaustion. | `ct5lp-hightpu-8t`<br>`ct6e-standard-8t` |
| **Tier 2 / 3** | **Sparse MoE & Dynamic Routing** | DeepSeek v4.1 Flash / MoE | Multi-Head Latent Attention (MLA) and sparse expert routing require Pallas MoE static capacity buffers and ICI All-to-All line-rate transfers. | `ct6e-standard-8t`<br>`ct5p-hightpu-4t/8t` |


