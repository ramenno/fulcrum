# fulcrum: Open-Weight Model Sizing, Serving & Management on Google Cloud Platform

[![Agent Skills Standard](https://img.shields.io/badge/Agent_Skills-v2.0-blue.svg)](https://agentskills.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Google Cloud Platform](https://img.shields.io/badge/GCP-Vertex%20AI%20%7C%20GCE%20%7C%20GKE-orange.svg)](https://cloud.google.com/)
[![Serving Engine](https://img.shields.io/badge/Engine-vLLM%20%28GPU%20%26%20TPU%29-green.svg)](https://github.com/vllm-project/vllm)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20L4%2FA100%2FH100%20%7C%20Cloud%20TPU%20v5e%2Fv6e%2Fv5p-purple.svg)](https://cloud.google.com/tpu)
[![Licence](https://img.shields.io/badge/Licence-Apache--2.0-lightgrey.svg)](LICENSE)

> *"Give me a lever long enough and a fulcrum on which to place it, and I shall move the world."* — Archimedes

**Fulcrum** provides effortless leverage to discover, size, provision, monitor, optimise, and benchmark open-weight, downloadable, self-deployable models (e.g. Google Gemma, Meta Llama, Alibaba Qwen, DeepSeek, and Mistral) on **Google Cloud Platform (GCP)** across **NVIDIA GPUs** and **Cloud TPUs (v5e / v6e Trillium / v5p)**.

Built in strict compliance with the open [Agent Skills standard](https://agentskills.io/), Fulcrum serves both as an autonomous toolchain for AI coding agents and as a CLI suite for machine learning engineers. Grab a cup of fresh coffee ☕ and let Fulcrum handle the cloud infrastructure math, quota verifications, and deployment orchestration.

---

## Core Philosophy: Downloadable Open Weights vs Closed APIs

Fulcrum focuses **exclusively on open-weight models** whose checkpoints can be downloaded, inspected, self-hosted, and served on user-controlled GCP infrastructure.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       WHAT FULCRUM SUPPORTS & MANAGES                       │
│                                                                             │
│  ✓ Downloadable weights: Hugging Face safetensors, GGUF, AWQ, FP8           │
│  ✓ Open architectures: Gemma 2/3, Llama 3.1/3.3, Qwen 2.5/3, DeepSeek R1    │
│  ✓ High-performance engines: vLLM (GPU) & vLLM-TPU with Pallas PagedAttention│
│  ✓ Full privacy & isolation: Zero prompt data egress outside your GCP VPC   │
│  ✓ Predictable economics: Hourly compute pricing (including Spot/Preemptible)│
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │ STRICT BOUNDARY
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXCLUDED FROM FULCRUM SCOPE                         │
│                                                                             │
│  ✗ Closed proprietary model APIs (e.g. OpenAI GPT-4o, Anthropic Claude)     │
│  ✗ Multi-tenant black-box SaaS APIs (e.g. Gemini API / Pay-per-token SaaS)  │
│  ✗ Proprietary cloud wrappers with uninspectable model weights              │
└─────────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Strictly Downloadable**: Fulcrum will **never** recommend, configure, or route requests to proprietary closed API services. Prompts, weights, activations, and KV cache remain strictly within your Google Cloud project boundary.

---

## Key Features

- 🔍 **Hardware & VRAM Sizing (`/fulcrum find`)**: Calculates exact weight footprints, KV cache allocation, and activation overheads across BF16, FP8, AWQ, and INT4 precision formats. Maps footprints to NVIDIA GPUs and Cloud TPU topologies.
- 🚀 **Vertex AI First Orchestration (`/fulcrum run`)**: Provisions **Vertex AI Endpoints** as the primary default using official Google Cloud vLLM container images. Automatically falls back to Compute Engine (or GKE) when Vertex is not viable.
- 🛡️ **Pre-Flight Quota Auditing**: Verifies regional GPU (`NVIDIA_L4_GPUS`, `NVIDIA_A100_GPUS`) and Cloud TPU (`custom_model_serving_tpu_v5e_chips`, `custom_model_serving_tpu_v6e_chips`) quota prior to provisioning, preventing `RESOURCE_EXHAUSTED` deployment rejections.
- 📊 **Live Telemetry & State Monitoring (`/fulcrum status`)**: Scrapes vLLM Prometheus metrics (GPU/HBM KV cache utilisation %, request queue depth, active requests) and reports endpoint lifecycle state.
- ⚡ **Benchmarking & Latency Sweeps (`/fulcrum benchmark`)**: Measures Time To First Token (TTFT), Time Per Output Token (TPOT), inter-token latency (ITL jitter), throughput, and concurrency saturation knees.
- 💬 **Interactive Streaming Web Chat (`/fulcrum chat`)**: Single-file web interface featuring Server-Sent Events (SSE) token streaming, real-time performance telemetry heads-up display (HUD), and automatic Google Cloud IAM access token lifecycle management.
- 🛠️ **Inference Flag Optimisation (`/fulcrum optimise`)**: Advises on chunked prefill, prefix caching, native FP8, tensor parallelism, and Pallas PagedAttention flags tailored to specific hardware architectures.
- 🧹 **Safe Teardown (`/fulcrum delete`)**: Deprovisions Vertex AI Endpoints, undeploys models, and removes Compute Engine VMs and firewall rules without leaving lingering cloud costs.

---

## Project Architecture

```
fulcrum/
├── fulcrum/
│   ├── SKILL.md                          # Main Agent Skills definition & instructions
│   ├── scripts/
│   │   ├── find_model.py                 # Model discovery & VRAM/HBM calculator
│   │   ├── launch_gcp.py                 # Deployment orchestrator (Vertex, GCE, GKE, Terraform)
│   │   ├── status_deployment.py          # State monitor & live Prometheus metrics inspector
│   │   ├── benchmark_deployment.py       # Latency, throughput, and concurrency benchmark suite
│   │   ├── chat_web.py                   # SSE streaming web chat interface with IAM token manager
│   │   ├── optimise_deployment.py        # vLLM parameter advisor & formulation engine
│   │   ├── delete_deployment.py          # Safe deprovisioning & teardown tool
│   │   ├── check_quota.py                # Pre-flight regional GPU and Cloud TPU quota auditor
│   │   ├── db.py                         # Local deployments database manager (deployments.json)
│   │   └── test_fulcrum.py               # Complete test verification suite
│   ├── references/
│   │   ├── models_guide.md               # Taxonomy of open-weight models, licences & memory math
│   │   ├── gcp_hardware_matrix.md        # GCP GPU & TPU instance matrix & quota catalogue
│   │   └── vllm_tuning_guide.md          # vLLM tuning runbook, XLA lifecycle & Cloud TPU guide
│   └── assets/
│       ├── eval_queries.json             # Agent trigger evaluation query dataset
│       └── templates/
│           ├── startup-script.sh         # Production GCE GPU startup script
│           ├── docker-compose.yaml       # Standalone Docker Compose vLLM deployment
│           └── terraform/main.tf         # Terraform infrastructure-as-code template
├── .agents/skills/fulcrum                # Workspace skill symlink (Agent Skills standard)
├── README.md                             # Project documentation
└── .venv/                                # Isolated Python virtual environment
```

---

## Getting Started

### Prerequisites

1. **Python 3.10+**: All scripts must be run inside a dedicated Python virtual environment (`.venv`).
2. **Google Cloud SDK (`gcloud`)**: Installed and configured on your machine.
3. **Google Cloud Project**: An active GCP project with appropriate compute and Vertex AI permissions.
4. **Hugging Face Token (`HF_TOKEN`)**: Required for gated open-weight models (e.g. Google Gemma 3, Meta Llama 3.1/3.3).

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/your-org/fulcrum.git
cd fulcrum

# 2. Create and activate a dedicated Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# 4. Set your active GCP project (Fulcrum auto-detects this)
gcloud config set project YOUR_GCP_PROJECT_ID

# 5. Export Hugging Face token (optional, for gated models)
export HF_TOKEN="hf_..."
```

---

## Subcommands & CLI Reference

All tools are executed through Python inside `.venv`:

### 1. `/fulcrum find` — Model Discovery & Memory Sizing

Calculates exact model parameter counts, weight memory footprints, and KV cache sizing across precisions, matching requirements to GCP hardware.

```bash
# Size for a 24 GiB NVIDIA L4 GPU
.venv/bin/python3 fulcrum/scripts/find_model.py --vram-limit 24

# Filter for coding models under permissive licences (Apache-2.0 / MIT)
.venv/bin/python3 fulcrum/scripts/find_model.py --task coding --license permissive

# Reasoning models running in native FP8 precision
.venv/bin/python3 fulcrum/scripts/find_model.py --task reasoning --precision fp8

# Discover models tailored for Cloud TPU hardware
.venv/bin/python3 fulcrum/scripts/find_model.py --query qwen --accelerator tpu
```

### 2. `/fulcrum run` — GCP Deployment Orchestrator

Provisions **Vertex AI Endpoints** as the primary default running official Google Cloud vLLM container images. Automatically performs pre-flight regional quota inspection and falls back to Compute Engine or GKE when Vertex is not viable.

```bash
# Primary Default: Dry-run preview of Vertex AI Endpoint deployment for Gemma 3 12B
.venv/bin/python3 fulcrum/scripts/launch_gcp.py --model google/gemma-3-12b-it --dry-run

# Provision on Cloud TPU v6e (4 chips) on Vertex AI
.venv/bin/python3 fulcrum/scripts/launch_gcp.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tpu-type v6e \
    --tpu-chips 4 \
    --dry-run

# Provision on Compute Engine (GCE fallback) with Spot pricing
.venv/bin/python3 fulcrum/scripts/launch_gcp.py \
    --model google/gemma-3-12b-it \
    --platform gce \
    --spot \
    --dry-run

# Stage weights in a regional Cloud Storage bucket (recommended for Cloud TPUs)
.venv/bin/python3 fulcrum/scripts/launch_gcp.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tpu-type v6e \
    --tpu-chips 4 \
    --gcs-model-path gs://your-bucket-us-central1/models/qwen14b \
    --dry-run

# Generate Terraform HCL infrastructure-as-code
.venv/bin/python3 fulcrum/scripts/launch_gcp.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --format terraform

# Directly execute provisioning on Google Cloud Platform
.venv/bin/python3 fulcrum/scripts/launch_gcp.py --model google/gemma-3-12b-it --execute
```

### 3. `/fulcrum status` — Deployment Telemetry & Utilisation

Monitors running endpoints and VMs, tests health routes (`/health`), and scrapes live vLLM Prometheus metrics:

```bash
# Inspect all tracked deployments (Vertex AI and GCE)
.venv/bin/python3 fulcrum/scripts/status_deployment.py

# Detailed diagnostics for a specific deployment ID
.venv/bin/python3 fulcrum/scripts/status_deployment.py --id fulcrum-gemma-3-12b-it --verbose

# Continuous live monitoring (poll every 5 seconds)
.venv/bin/python3 fulcrum/scripts/status_deployment.py --watch 5
```

### 4. `/fulcrum benchmark` — Latency & Throughput Benchmarking

Conducts comprehensive load testing across TTFT, TPOT, inter-token jitter, and concurrency sweeps:

```bash
# Run a standard benchmark against an active deployment
.venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --id fulcrum-gemma-3-12b-it

# Concurrency sweep (1, 2, 4, 8, 16 concurrent streams) to locate saturation
.venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --id fulcrum-gemma-3-12b-it --sweep

# Export benchmark results as a formatted Markdown report
.venv/bin/python3 fulcrum/scripts/benchmark_deployment.py \
    --id fulcrum-gemma-3-12b-it \
    --report-file benchmark_report.md
```

### 5. `/fulcrum chat` — Interactive Streaming Web Chat

Launches a single-file web interface with real-time token streaming and an inference telemetry heads-up display:

```bash
# Launch interactive chat for an active deployment
.venv/bin/python3 fulcrum/scripts/chat_web.py --id fulcrum-gemma-3-12b-it --port 8080

# Launch in simulated mock mode (ideal for UI testing without active cloud costs)
.venv/bin/python3 fulcrum/scripts/chat_web.py --mock --port 8080
```

### 6. `/fulcrum optimise` — Inference Parameter Advisor

Formulates mathematically sound vLLM serving parameters for your target hardware and workload profile (latency, throughput, or balanced):

```bash
# High-throughput parameter formulation for NVIDIA L4
.venv/bin/python3 fulcrum/scripts/optimise_deployment.py advise \
    --model google/gemma-3-12b-it \
    --gpu-type nvidia-l4 \
    --workload throughput

# Performance tuning advice for Cloud TPU v6e (Trillium)
.venv/bin/python3 fulcrum/scripts/optimise_deployment.py advise \
    --model Qwen/Qwen2.5-14B-Instruct \
    --gpu-type ct6e-standard-4t \
    --gpu-count 4
```

### 7. `/fulcrum delete` — Safe Teardown & Deprovisioning

Deprovisions cloud resources, deletes Vertex endpoints, tears down GCE VMs, and cleans up firewall rules:

```bash
# Preview deletion commands for a deployment (dry-run)
.venv/bin/python3 fulcrum/scripts/delete_deployment.py --id fulcrum-gemma-3-12b-it

# Execute teardown and remove local database record
.venv/bin/python3 fulcrum/scripts/delete_deployment.py --id fulcrum-gemma-3-12b-it --execute

# Deprovision all tracked deployments created by Fulcrum
.venv/bin/python3 fulcrum/scripts/delete_deployment.py --all --execute
```

### 8. `/fulcrum quota` — Pre-Flight Accelerator Quota Auditor

Audits available quota headroom across Vertex AI and Compute Engine, with automated regional sweeping:

```bash
# Proactively sweep all candidate TPU regions for Cloud TPU v6e (Trillium)
.venv/bin/python3 fulcrum/scripts/check_quota.py --is-tpu --tpu-type v6e --count 8 --sweep

# Verify regional NVIDIA L4 quota on Vertex AI
.venv/bin/python3 fulcrum/scripts/check_quota.py --platform vertex --accelerator nvidia-l4 --count 1 --region us-central1

# Audit Cloud TPU quota for a specific region
.venv/bin/python3 fulcrum/scripts/check_quota.py --platform vertex --accelerator v6e --count 4 --region europe-west4 --is-tpu
```

---

## Architectural Rules & Hardware Invariants

When deploying open-weight models on GCP, Fulcrum enforces critical architectural invariants:

### 1. Platform Selection Hierarchy
1. **Vertex AI Prediction Endpoints (Primary Default)**: Single-node GPU and Cloud TPU workloads run on Vertex AI. Advantages include Google Cloud IAM token authentication, zero VM patch management, automatic container health checking, and scalable replica control.
2. **Compute Engine (Fallback)**: Used when custom host kernel modules are required, or when unauthenticated direct VPC port access is necessary.
3. **Google Kubernetes Engine (Fallback)**: Used for multi-node distributed serving (>8 GPUs or multi-slice TPUs).

### 2. G2 Rigid Hardware Topology
Google Cloud enforces strict 1:1 hardware topology on NVIDIA L4 (G2) instances. Fulcrum guarantees correct machine pairing:
- `1x NVIDIA L4` ➔ `g2-standard-8` (32 GiB host RAM)
- `2x NVIDIA L4` ➔ `g2-standard-24` (96 GiB host RAM). *Note: `g2-standard-16` is strictly rejected by the GCP API.*
- `4x NVIDIA L4` ➔ `g2-standard-48` (192 GiB host RAM)
- `8x NVIDIA L4` ➔ `g2-standard-96` (384 GiB host RAM)

### 3. Cloud TPU Vertex AI Invariants
Cloud TPUs operate under a different serving paradigm than GPUs:
- **No `--accelerator` Flag**: Cloud TPU machine types (`ct5lp-hightpu-*`, `ct6e-standard-*`, `ct5p-hightpu-*`) have hardwired TPU chips. Passing `--accelerator=type=...,count=...` triggers immediate API rejection (`INVALID_ARGUMENT: Accelerator type must not be specified for TPU machines`).
- **No `--swap-space` in TPU Builds**: CPU RAM swapping for KV cache offloading (`--swap-space`) is unsupported in vLLM TPU builds (`vllm.entrypoints.openai.api_server`) and triggers immediate container exit code 2:
  ```text
  api_server.py: error: unrecognized arguments: --swap-space=8
  ```
  On Cloud TPUs, KV caching is managed entirely within High Bandwidth Memory (HBM) via Pallas PagedAttention (`VLLM_TPU_ENABLE_PALLAS=1`).
- **Dedicated Resources**: Cloud TPUs require fixed replica counts (`--min-replica-count=1 --max-replica-count=1`).
- **High Shared Memory & Extended Timeout**: Cloud TPUs require `--container-shared-memory-size-mb=32768` (32 GiB) for PyTorch/XLA POSIX shm and `--container-deployment-timeout-seconds=7200` (2 hours) to accommodate Ahead-of-Time (AOT) XLA graph compilation.
- **HBM Utilisation**: Set `--gpu-memory-utilization=0.85` to preserve HBM headroom for XLA compilation buffers.
- **Regional GCS Weight Staging**: Stage weights in a regional Cloud Storage bucket (`GCS_MODEL_PATH`) co-located with the endpoint region to avoid Hugging Face rate limits and container download timeouts.

### 4. Proactive Multi-Region TPU Quota Sweeping
Cloud TPU allocations differ significantly across GCP regions. For example, a project may have ample GPU quota in `us-central1`, but zero quota for `custom_model_serving_tpu_v6e` in `us-central1` while holding an active 8-chip allocation in `europe-west4`.
When `--tpu-type` or `--accelerator tpu` is specified without an explicit `--region`:
- Fulcrum automatically queries `aiplatform.googleapis.com` via Cloud Quotas across all known candidate TPU regions (`europe-west4`, `us-central1`, `asia-southeast1`, `us-east4`).
- It ranks candidate regions by available headroom.
- It dynamically routes provisioning to the region with verified capacity (e.g. automatically targeting `europe-west4`).
- Explicit regional parameters (`--region <region>`) are strictly honoured.

### 5. Layer 3 CUDA Graph Guardrail
On large models (≥20B parameters) served on constrained VRAM (≤2x L4), vLLM's Ahead-of-Time CUDA graph memory capture allocates ~5–6 GiB per GPU, risking an immediate `OutOfMemoryError`. Fulcrum automatically enables `--enforce-eager` and injects `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reclaim VRAM.

---

## State Management

Fulcrum maintains an idempotent local state database tracking all active and past deployments at:
- `~/.config/fulcrum/deployments.json` (or `.fulcrum/deployments.json` as workspace fallback)
- Customisable via the `FULCRUM_DB_PATH` environment variable.

You can inspect the state file directly:
```bash
.venv/bin/python3 fulcrum/scripts/db.py
```

---

## Running the Test Suite

Fulcrum features an end-to-end automated verification suite covering all 9 modules:

```bash
# Run the complete test suite within the virtual environment
.venv/bin/python3 fulcrum/scripts/test_fulcrum.py
```

The test runner verifies:
- `db.py`: CRUD operations, file locking, and project auto-detection.
- `find_model.py`: Memory calculations, licensing filters, and TPU/GPU hardware mapping.
- `launch_gcp.py`: Vertex AI commands, GCE scripts, GKE manifests, quota checks, and Terraform HCL generation.
- `status_deployment.py`: Metric scraping and health evaluation.
- `optimise_deployment.py`: Parameter formulation and TPU/GPU flag validation.
- `delete_deployment.py`: Safe deprovisioning and state updates.
- `benchmark_deployment.py`: Prefill, decode, and sweep calculations.
- `chat_web.py`: SSE endpoints and HTML interface generation.
- `check_quota.py`: Regional GPU and Cloud TPU quota verification.

---

## Technical Documentation & References

For comprehensive technical deep-dives, consult the references directory:
- [models_guide.md](fulcrum/references/models_guide.md): Open-weight taxonomy, architecture comparisons, and mathematical memory sizing formulas.
- [gcp_hardware_matrix.md](fulcrum/references/gcp_hardware_matrix.md): Complete matrix of NVIDIA GPUs and Cloud TPUs, regional availability, pricing, and quota metrics.
- [vllm_tuning_guide.md](fulcrum/references/vllm_tuning_guide.md): Advanced performance tuning runbook, chunked prefill, prefix caching, XLA compilation, and Cloud TPU runbooks.

---

## Licence & Compliance

Fulcrum is released under the **Apache-2.0 Licence** and conforms to the [Agent Skills standard](https://agentskills.io/).
