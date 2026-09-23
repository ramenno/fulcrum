---
name: fulcrum
description: Fulcrum provides effortless leverage to discover, size, provision, monitor, optimise, and benchmark open-weight, downloadable, self-deployable models (e.g. Gemma, Llama, Qwen, DeepSeek, Mistral) on GCP (NVIDIA GPUs & Cloud TPUs v5e/v6e/v5p), excluding closed API-based access. Calculates VRAM/HBM and hardware footprints, provisions Vertex AI Endpoints as primary default (with Compute Engine or GKE fallback when not viable) using vLLM, monitors status and live utilisation metrics, optimises inference flags, and safely deprovisions resources. Use whenever the user asks to find, size, deploy, monitor, benchmark, or delete self-hosted models on GCP.
compatibility: Requires Python 3.10+, gcloud CLI, and access to GCP project with GPU or TPU quota.
metadata:
  version: "2.0.0"
  author: "Antigravity & Pair Programmer"
  category: "Machine Learning & Cloud Infrastructure"
---

# /fulcrum: Effortless Leverage for Open-Weight Models on GCP

`/fulcrum` is an end-to-end skill dedicated to discovering open-weight models, sizing compute, GPU VRAM, and Cloud TPU HBM memory requirements, orchestrating automated deployments on Google Cloud Platform (GCP) using vLLM, inspecting live performance and utilisation metrics, fine-tuning serving configurations, and safely deprovisioning infrastructure.

## Core Philosophy: Downloadable Weights vs API-Based Access

`/fulcrum` focuses **strictly on open-weight, downloadable, and self-deployable models** that developers can download, inspect, self-host, and run on their own hardware.

- **In Scope (Self-Hosted Open Weights)**:
  - Downloadable model weights published on repositories like Hugging Face (e.g. Kimi, GLM, Qwen & Qwen Coder, DeepSeek R1 & distillations etc).
  - Checkpoints stored as `.safetensors`, GGUF, or AWQ/FP8 formats.
  - Self-hosted inference engines (vLLM, SGLang, TGI) serving on user-managed GCP infrastructure (NVIDIA GPUs and Cloud TPUs).
  - Complete control over model weights, parameters, data privacy, and serving infrastructure.

- **Explicitly Out of Scope (Proprietary & API-Only Services)**:
  - Closed proprietary model APIs (e.g. OpenAI GPT-4o, Anthropic Claude 3.5, Google Gemini API).
  - Managed SaaS pay-per-token API endpoints where weights cannot be inspected or downloaded.
  - Proprietary cloud-hosted API wrappers.
  - `/fulcrum` **never** recommends, configures, or routes traffic to third-party closed API services.

---

## Subcommands Overview

The skill provides seven specialised workflows:

| Subcommand | Description | Primary Helper Script |
| :--- | :--- | :--- |
| `/fulcrum find` | Search downloadable open-weight models by capability, license, or size; calculate exact VRAM/HBM; select GCP GPU or Cloud TPU instances. | `scripts/find_model.py` |
| `/fulcrum run` | Provision **Vertex AI Endpoints** as primary default (or GCE/GKE fallback) running official vLLM / vLLM-TPU container, and record in state database. | `scripts/launch_gcp.py` |
| `/fulcrum status` | Inspect Vertex AI Endpoint and GCE VM lifecycle states, health, live KV cache utilisation, and previous benchmark scores. | `scripts/status_deployment.py` |
| `/fulcrum benchmark` | Measure prefill latency (TTFT), decode latency (TPOT/ITL), jitter, throughput, and multi-stream concurrency saturation. | `scripts/benchmark_deployment.py` |
| `/fulcrum chat` | Interactive web front end for streaming model chat with managed Google Cloud IAM token and live inference stats (TTFT, tok/s, TPOT). | `scripts/chat_web.py` |
| `/fulcrum optimise` | Compute optimal vLLM runtime parameters (FP8, Pallas PagedAttention, chunked prefill, prefix caching) tailored to target GPU/TPU hardware. | `scripts/optimise_deployment.py` |
| `/fulcrum delete` | Safely deprovision Vertex AI Endpoints, GCE instances, and firewall rules created by `/fulcrum run`, updating the local registry. | `scripts/delete_deployment.py` |

---

## 1. /fulcrum find — Downloadable Model Discovery & Hardware Sizing

Use `/fulcrum find` when the user asks for open-weight model recommendations, memory calculations, or hardware sizing for self-hosting.

> [!NOTE]
> `/fulcrum find` only queries and evaluates models with publicly downloadable weights. It does not recommend proprietary API-only models.

### Workflow:
1. **Identify User Requirements**:
   - Primary task: `general`, `coding`, `reasoning`, `math`, `multilingual`, `agentic`, or `edge`.
   - Accelerator preference: `--accelerator [all|gpu|tpu]`. Cloud TPUs (v5e, v6e Trillium, v5p) offer 45–58% cost savings for compatible dense architectures.
   - Size/Compute constraints: maximum parameter count (e.g. 14B, 32B, 70B) or maximum VRAM/HBM.
   - License constraints: `permissive` (Apache-2.0 / MIT) or open-weights (Gemma / Llama).
   - Precision: `bf16`, `fp8`, `awq`, or `int4`.
2. **Execute Discovery Tool**:
   Run `scripts/find_model.py` using the Python virtual environment:
   ```bash
   # Find models with both GPU and TPU recommendations
   .venv/bin/python3 fulcrum/scripts/find_model.py --task <task> --precision <precision> --license <license>

   # Filter specifically for Cloud TPU deployment recommendations
   .venv/bin/python3 fulcrum/scripts/find_model.py --query qwen --accelerator tpu
   ```
3. **Present Recommendation**:
   - Display the recommended downloadable model, strengths, memory breakdown (Weights + KV Cache + Headroom), and recommended GCP GPU and Cloud TPU instances.
   - For in-depth model architecture information, consult [references/models_guide.md](references/models_guide.md).

### Examples:
```bash
# Find downloadable coding models under Apache-2.0 that fit on a single 24GB L4 GPU
.venv/bin/python3 fulcrum/scripts/find_model.py --task coding --license permissive --vram-limit 24

# Sizing for self-hostable models on Cloud TPU
.venv/bin/python3 fulcrum/scripts/find_model.py --query gemma --accelerator tpu
```

---

## 2. /fulcrum run — GCP Deployment Orchestration

Use `/fulcrum run` to provision and launch a downloadable open-weight model on GCP.

### Platform Priority Hierarchy:
1. **Vertex AI Endpoint (Primary Default)**:
   - Evaluated first whenever viable for both NVIDIA GPUs and Cloud TPUs.
   - For GPUs: Uses official Google Cloud vLLM container (`pytorch-vllm-serve:latest`).
   - For Cloud TPUs (`v5e`, `v6e`, `v5p`): Uses `vllm/vllm-tpu:latest` on `ct5lp-*`, `ct6e-*`, or `ct5p-*` machines with Pallas PagedAttention and static bucketing.
   - Zero VM management, automated health probing, and IAM token-based authentication.
2. **Compute Engine (GCE) Fallback**:
   - Selected when Vertex AI is not viable (e.g., custom host OS kernel modules, unauthenticated public port binding, local RAID NVMe scratch) or explicitly requested via `--platform gce`.
3. **Kubernetes Engine (GKE) Fallback**:
   - Selected when models require multi-node distributed serving (>8 GPUs / multi-slice TPUs) or explicitly requested via `--platform gke`.

### Workflow:
1. **Check Prerequisites & GCP Project Resolution**:
   - Determine the active GCP project by running `gcloud config get-value project`. If a project is configured, use it automatically.
   - **Only ask the user if there is no project set** (e.g. `(unset)` or blank).
   - If deploying gated models (e.g. Meta Llama or Google Gemma), supply `HF_TOKEN` so the container can download weights from Hugging Face.
2. **Proactive Pre-Flight Quota Verification**:
   - `/fulcrum run` automatically audits GPU/TPU quota headroom prior to provisioning.
   - You can also run the quota auditor standalone:
     ```bash
     # Verify GPU quota
     .venv/bin/python3 fulcrum/scripts/check_quota.py --platform vertex --accelerator nvidia-l4 --count 1 --region us-central1

     # Verify Cloud TPU quota
     .venv/bin/python3 fulcrum/scripts/check_quota.py --platform vertex --accelerator v6e --count 4 --region us-central1 --is-tpu
     ```
   - If quota is insufficient, `/fulcrum run` highlights available headroom, suggests alternative regions (`us-central1`, `europe-west4`), and provides the direct GCP Console URL to request quota increases. Pass `--strict-quota` to abort immediately on missing quota.
3. **Generate Deployment Plan (Dry-Run)**:
   By default, `/fulcrum run` automatically targets Vertex AI Endpoints:
3. **Direct Provisioning & Execution**:
   When the necessary deployment parameters are known (model identifier, platform, quantisation, and GCP project), `/fulcrum run` proceeds immediately with execution on Google Cloud Platform without requiring confirmation prompts or an `--execute` flag:
   ```bash
   # Standard GPU deployment on Vertex AI
   # Standard GPU deployment on Vertex AI (executes immediately)
   .venv/bin/python3 fulcrum/scripts/launch_gcp.py --model <huggingface-id> [--project <project>] [--region <region>]

   # Cloud TPU deployment on Vertex AI (e.g. TPU v6e Trillium 4 chips)
   .venv/bin/python3 fulcrum/scripts/launch_gcp.py --model <huggingface-id> --tpu-type v6e --tpu-chips 4

   # Cloud TPU deployment with regional GCS bucket weight staging
   .venv/bin/python3 fulcrum/scripts/launch_gcp.py --model <huggingface-id> --tpu-type v6e --tpu-chips 4 --gcs-model-path gs://my-bucket/models/qwen

   # Explicit override to Compute Engine fallback
   .venv/bin/python3 fulcrum/scripts/launch_gcp.py --model <huggingface-id> --platform gce
   ```
   Add `--record` to register the planned deployment into the local state database immediately.
4. **Execute Provisioning**:
   When approved, execute provisioning directly with `--execute`:
4. **Dry-Run & Infrastructure-as-Code Generation**:
   If the user specifically requests a preview of deployment commands without provisioning infrastructure, or requests Terraform/JSON definitions, pass `--dry-run` or `--format`:
   ```bash
   .venv/bin/python3 fulcrum/scripts/launch_gcp.py --model <huggingface-id> --execute
   # Dry-run inspection without provisioning
   .venv/bin/python3 fulcrum/scripts/launch_gcp.py --model <huggingface-id> --dry-run

   # Generate Terraform HCL
   .venv/bin/python3 fulcrum/scripts/launch_gcp.py --model <huggingface-id> --format terraform
   ```
   Add `--record` to register the planned deployment into the local state database in dry-run mode.
5. **Hardware Reference**:
   Consult [references/gcp_hardware_matrix.md](references/gcp_hardware_matrix.md) for Vertex machine mappings (GPUs and TPUs) and pricing comparisons.

---

## 3. /fulcrum status — Deployment Monitoring & Utilisation

Use `/fulcrum status` to check the status, health, and live utilisation metrics of models deployed by `/fulcrum run`.

### Workflow:
1. **Query Deployment Registry**:
   Run `scripts/status_deployment.py` to inspect all tracked deployments:
   ```bash
   .venv/bin/python3 fulcrum/scripts/status_deployment.py
   ```
2. **Detailed Diagnostics & Metrics**:
   Inspect platform specifics (Vertex prediction URI, IAM OAuth2 curl snippet, or GCE KV cache Prometheus load):
   ```bash
   .venv/bin/python3 fulcrum/scripts/status_deployment.py --id <resource-name> --verbose
   ```
3. **Continuous Monitoring**:
   Use `--watch <seconds>` to monitor live traffic and memory utilisation continuously.

### Metrics Monitored:
- **Platform**: `VERTEX` vs `GCE`.
- **Lifecycle Status**: Vertex Endpoint state (`ACTIVE`, `NO_MODELS`) or GCE VM status (`RUNNING`, `TERMINATED`).
- **Health**: Service health probe against `/health`.
- **KV Cache & Concurrency**: Scraped from vLLM Prometheus metrics for GCE instances.

---

## 4. /fulcrum benchmark — Live Model Performance Benchmarking

Use `/fulcrum benchmark` to generate empirical serving benchmarks for deployed models. Unlike closed API benchmarks, `/fulcrum benchmark` evaluates metrics critical to self-hosted LLM engines: prefill latency (TTFT), decode latency (TPOT / ITL), inter-token arrival jitter, and multi-stream concurrency saturation curves.

### Workload Profiles:
- `chat`: Balanced conversational queries (~150 prompt tokens, 150 max output tokens).
- `rag`: Long-context prefill stress (~1,000 prompt tokens, 80 max output tokens) to test prompt attention and chunked prefill efficiency.
- `code` / `reasoning`: Decode-heavy code generation (~80 prompt tokens, 450 max output tokens) to test autoregressive decode memory bandwidth.
- `custom`: Custom prompt and max token configurations.

### Key Metrics Captured:
- **TTFT (Prefill Latency)**: Min, P50, P90, P95, P99, Max, and Mean.
- **TPOT (Decode Latency / ITL)**: Min, P50, P90, P95, P99, Max, and Mean.
- **Inter-Token Jitter**: Standard deviation of per-token arrival delays.
- **Throughput**: Generation throughput (output tokens/sec), prompt throughput (prefill tokens/sec), and request throughput (RPS).
- **Concurrency Scaling (`--sweep`)**: Automated multi-concurrency sweep (1, 2, 4, 8, 16 streams) detecting the saturation knee and queue penalties.
- **Telemetry**: KV cache usage factor and queue depth when accessible.

### Workflow:
1. **Benchmark by Deployment ID**:
   ```bash
   # Benchmark deployment tracked in state database
   .venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --id fulcrum-qwen3-8-27b

   # Test RAG prefill performance under concurrency
   .venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --id fulcrum-qwen3-8-27b --workload rag --concurrency 4
   ```
2. **Automated Concurrency Saturation Sweep**:
   ```bash
   # Identify optimal concurrency knee and saturation point
   .venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --id fulcrum-qwen3-8-27b --sweep
   ```
3. **Benchmark Direct Endpoints (Vertex AI or GCE)**:
   ```bash
   # Direct Vertex AI Endpoint (IAM token auto-resolved via gcloud)
   .venv/bin/python3 fulcrum/scripts/benchmark_deployment.py \
       --endpoint https://us-central1-aiplatform.googleapis.com/v1/projects/<PROJECT>/locations/us-central1/endpoints/<ID>:rawPredict \
       --model Qwen/Qwen3.8-27B --concurrency 4

   # Direct GCE / Local instance
   .venv/bin/python3 fulcrum/scripts/benchmark_deployment.py \
       --endpoint http://<INSTANCE_IP>:8000/v1 --model google/gemma-3-12b-it
   ```
4. **Export Reports**:
   Export findings as JSON or Markdown report files:
   ```bash
   .venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --id fulcrum-qwen3-8-27b --format json
   .venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --id fulcrum-qwen3-8-27b --report-file benchmark_report.md
   ```

---

## 5. /fulcrum chat — Interactive Web Chat & Real-Time Telemetry

Use `/fulcrum chat` when the user wants to test, interact with, or demonstrate a deployed model in a live browser interface with real-time token streaming and inference telemetry.

### Features & Architecture:
- **Managed Google Cloud IAM Authentication**: Operates as a Backend-For-Frontend (BFF), securely handling `gcloud auth print-access-token` caching and refresh server-side without exposing credentials to the client.
- **Real-Time Token Streaming**: Streams token deltas via standard Server-Sent Events (SSE) with live typing indicators.
- **Live Inference Stats HUD**: Surfaces real-time metrics beneath every message:
  - ⏱️ **TTFT**: Time To First Token in milliseconds.
  - ⚡ **Throughput**: Generation speed in output tokens/second (tok/s).
  - 🔢 **Token Count**: Total tokens generated for the response.
  - ⏳ **TPOT**: Time Per Output Token (decode latency) in milliseconds.
  - 🕒 **Turnaround Time**: Total end-to-end request latency.
- **Model Switching**: Dynamically displays all active deployments recorded in `deployments.json` in an interactive dropdown selector.

### Workflow:
1. **Launch Chat with Active Deployment**:
   ```bash
   # Automatically targets latest deployment in deployments.json and opens browser
   .venv/bin/python3 fulcrum/scripts/chat_web.py

   # Target a specific deployment by ID
   .venv/bin/python3 fulcrum/scripts/chat_web.py --id fulcrum-qwen3-8-27b
   ```
2. **Connect to Direct Endpoint**:
   ```bash
   # Connect to custom Vertex AI Endpoint
   .venv/bin/python3 fulcrum/scripts/chat_web.py \
       --endpoint https://us-central1-aiplatform.googleapis.com/v1/projects/<PROJECT>/locations/us-central1/endpoints/<ID>:rawPredict \
       --model Qwen/Qwen3.8-27B
   ```
3. **Headless / Offline Verification**:
   ```bash
   # Run offline with synthetic streaming
   .venv/bin/python3 fulcrum/scripts/chat_web.py --mock --no-browser --port 8085
   ```

---

## 6. /fulcrum optimise — Serving Parameter Formulation & Tuning

Use `/fulcrum optimise` to configure vLLM serving parameters for maximum performance on self-hosted GPU and Cloud TPU instances.

### Workflow:
1. **Offline Parameter Formulation (`advise`)**:
   Calculate architecture-specific parameters based on accelerator type (GPU or TPU) and workload target:
   ```bash
   # GPU tuning formulation
   .venv/bin/python3 fulcrum/scripts/optimise_deployment.py advise \
       --model <huggingface-id> \
       --gpu-type <nvidia-l4|nvidia-tesla-a100|nvidia-h100-80gb> \
       --workload <throughput|latency|balanced> \
       --concurrency <expected-streams>

   # Cloud TPU tuning formulation (e.g. TPU v6e Trillium 4 chips)
   .venv/bin/python3 fulcrum/scripts/optimise_deployment.py advise \
       --model <huggingface-id> \
       --gpu-type ct6e-standard-4t \
       --gpu-count 4 \
       --workload balanced
   ```
2. **Deep-Dive Reference**:
   For detailed parameter mechanics (PagedAttention, Pallas KV cache, static shape bucketing, chunked prefill, prefix caching, CUDA graph and XLA memory budgeting), consult [references/vllm_tuning_guide.md](references/vllm_tuning_guide.md).

---

## 7. /fulcrum delete — Deprovisioning & Cleanup

Use `/fulcrum delete` when the user wants to tear down self-hosted instances, Vertex AI Endpoints, and firewall rules to eliminate cloud costs.

### Workflow:
1. **Preview Teardown Plan (Dry-Run)**:
   Always preview the resources to be destroyed before executing:
   ```bash
   # Preview deletion of a specific deployment
   .venv/bin/python3 fulcrum/scripts/delete_deployment.py --id <resource-name>

   # Preview teardown of all tracked deployments
   .venv/bin/python3 fulcrum/scripts/delete_deployment.py --all
   ```
2. **Execute Cleanup**:
   Confirm with `--execute` to permanently undeploy Vertex models and delete endpoints, or delete GCE GPU instances, release firewall rules, and prune the local database:
   ```bash
   .venv/bin/python3 fulcrum/scripts/delete_deployment.py --id <resource-name> --execute
   ```

---

## Gotchas & Critical Rules

- **GCP Project Auto-Detection**: Always find the GCP project by running `gcloud config get-value project` and use that project automatically. Only ask the user if there is no project set in `gcloud`.
- **Vertex Endpoint First Priority**: `/fulcrum run` always attempts to provision a Vertex AI Endpoint first. Only fall back to Compute Engine (or GKE) if Vertex is not viable (e.g. multi-node >8 GPUs, custom host kernel requirements, or missing quota).
- **Proactive Accelerator Quota Auditing**: Never initiate provisioning without verifying regional GPU (`NVIDIA_L4_GPUS`, `NVIDIA_A100_GPUS`) or Cloud TPU (`custom_model_serving_tpu_v5e_chips`, `custom_model_serving_tpu_v6e_chips`) quota. Insufficient quota causes immediate `RESOURCE_EXHAUSTED` deployment rejections. Check quota beforehand with `.venv/bin/python3 fulcrum/scripts/check_quota.py` or rely on `/fulcrum run`'s automatic pre-flight check.
- **Cloud TPU Vertex AI Rules**:
  - **No Accelerator Flag**: Cloud TPU machine types (`ct5lp-hightpu-*`, `ct6e-standard-*`, `ct5p-hightpu-*`) have hardwired TPU topologies. **Never pass `--accelerator=type=...,count=...`** during `gcloud ai endpoints deploy-model` or Terraform `machine_spec`. Passing it causes immediate rejection (`INVALID_ARGUMENT: Accelerator type must not be specified for TPU machines`).
  - **No `--swap-space` Flag in TPU Builds**: CPU RAM swapping for KV cache offloading (`--swap-space`) is unsupported in vLLM Cloud TPU builds (`vllm/vllm-tpu` / `vllm.entrypoints.openai.api_server`) and causes Python argparse to abort with exit code 2 (`api_server.py: error: unrecognized arguments: --swap-space=8`). All KV blocks are managed directly in TPU HBM via Pallas PagedAttention. Never pass `--swap-space` for Cloud TPU deployments or optimisation advice.
  - **Dedicated Resources (No Autoscaling)**: Cloud TPUs require fixed replica counts: `--min-replica-count=1 --max-replica-count=1`.
  - **High Shared Memory & Extended Timeout**: Always configure `--container-shared-memory-size-mb=32768` (32 GiB) for PyTorch/XLA POSIX shm and `--container-deployment-timeout-seconds=7200` (2 hours) to accommodate Ahead-of-Time (AOT) XLA graph compilation.
  - **HBM Utilisation**: Set `--gpu-memory-utilization 0.85` on Cloud TPUs to preserve HBM headroom for XLA compilation buffers.
  - **Pallas PagedAttention**: Inject `PJRT_DEVICE=TPU` and `VLLM_TPU_ENABLE_PALLAS=1` for hardware-accelerated HBM KV cache management.
  - **Weight Staging**: Always stage weights in a regional Cloud Storage bucket (`GCS_MODEL_PATH`) co-located with the endpoint region to avoid Hugging Face rate limits and container download timeouts.
- **G2 Rigid Hardware Topology**: Google Cloud enforces strict 1:1 hardware topology on G2 instances. `g2-standard-4` through `g2-standard-32` allow strictly 1x L4; 2x L4 requires `g2-standard-24`; 4x L4 requires `g2-standard-48`; 8x L4 requires `g2-standard-96`. Never pair `g2-standard-16` with 2x L4.
- **Vertex Container Command Contract**: The Vertex Model Garden vLLM container (`pytorch-vllm-serve`) entrypoint `gcs_download_launcher.sh` requires explicit `--container-command="python3,-m,vllm.entrypoints.openai.api_server"` to prevent executable lookup failures (exit code 127).
- **CUDA Graph Memory Footprint on ≥20B Models**: vLLM Ahead-of-Time CUDA graph memory profiling allocates ~6.0 GiB per GPU. Models ≥27B (e.g. Qwen 27B) on G2 require 4x L4 (`g2-standard-48`) or 1x A100-80GB (`a2-ultragpu-1g`). If running on ≤2x L4, enable `--enforce-eager` to reclaim ~5–6 GiB VRAM per GPU and inject `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Strictly Open-Weight & Downloadable (No Closed APIs)**: This skill exclusively supports downloadable, self-deployable open-weight model checkpoints. Never advise or configure proprietary, closed API services (e.g. OpenAI, Anthropic, or SaaS API endpoints).
- **Authentication**: Vertex AI Endpoints require Google Cloud IAM OAuth2 tokens (`Authorization: Bearer $(gcloud auth print-access-token)`), whereas GCE exposes ports through VPC firewall tags.
- **Local State Database**: Tracked deployments are saved to `~/.config/fulcrum/deployments.json` (or `.fulcrum/deployments.json` if user configuration directories are restricted). Override via `FULCRUM_DB_PATH` (or `SOW_DB_PATH` fallback).
- **Shared Firewall Protection**: `/fulcrum delete` verifies whether other running deployments share the same port before attempting to delete firewall rules.
- **Gated Models & HF_TOKEN**: Gated open-weight models (such as `google/gemma-3-*` and `meta-llama/Llama-3.*`) require accepting terms on Hugging Face and providing a valid `HF_TOKEN` to download weights onto the node.
- **Native FP8 Compatibility**: Ada Lovelace (NVIDIA L4) and Hopper (NVIDIA H100) support native FP8 inference in vLLM. NVIDIA A100 (Ampere) does not have native FP8 Tensor Cores; use BF16 or AWQ for A100 deployments.
- **Python Virtual Environment**: Always execute Python scripts within `.venv` (`.venv/bin/python3 ...`) per user requirements.
