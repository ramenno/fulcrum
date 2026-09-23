# GCP GPU Hardware & Instance Matrix

This reference documents the GPU options, machine series, memory specifications, and platform targets available on Google Cloud Platform for serving open-weight models with vLLM.

---

## Target Platform Comparison

| Platform Target | Role & Priority | Management Model | Best Suited Scenarios |
| :--- | :--- | :--- | :--- |
| **Vertex AI Endpoint** | **Priority 1 (Default)** | Fully managed prediction service | Single-node serving (1 to 8 GPUs), managed health probing, IAM token security, zero host OS patching |
| **Compute Engine (GCE)** | **Priority 2 (Fallback)** | Self-managed GPU virtual machine | Direct public IP port binding, custom host OS/kernel modules, local NVMe scratch RAID, or when Vertex quota is unavailable |
| **Kubernetes Engine (GKE)** | **Priority 3 (Fallback)** | Container orchestration cluster | Multi-node distributed inference (>8 GPUs, e.g. 671B MoE clusters), custom Ray/vLLM Helm deployments |

---

## GPU Accelerator Overview

| GPU Model | Architecture | VRAM / Card | Interconnect | Native FP8 | Vertex Accelerator Type | GCE / GKE Accelerator | Hourly Cost (Approx) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA L4** | Ada Lovelace | 24 GB GDDR6 | PCIe Gen4 | Yes | `NVIDIA_L4` | `nvidia-l4` | ~$0.70 / GPU |
| **NVIDIA A100 (40GB)** | Ampere | 40 GB HBM2 | NVLink 3 (600 GB/s) | No | `NVIDIA_TESLA_A100` | `nvidia-tesla-a100` | ~$3.67 / GPU |
| **NVIDIA A100 (80GB)** | Ampere | 80 GB HBM2e | NVLink 3 (600 GB/s) | No | `NVIDIA_A100_80GB` | `nvidia-a100-80gb` | ~$5.00 / GPU |
| **NVIDIA H100 (80GB)** | Hopper | 80 GB HBM3 | NVLink 4 (900 GB/s) | Yes (4th Gen) | `NVIDIA_H100_80GB` | `nvidia-h100-80gb` | ~$10.50 / GPU |
| **NVIDIA T4** | Turing | 16 GB GDDR6 | PCIe Gen3 | No | `NVIDIA_TESLA_T4` | `nvidia-tesla-t4` | ~$0.35 / GPU |

---

## Vertex AI Endpoint Machine Configurations

Vertex AI Endpoints deploy models inside dedicated containers (`us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:latest`) attached to dedicated machine types:

### 1. G2 Series (NVIDIA L4)
- `g2-standard-8`: 1x `NVIDIA_L4` (24GB), 8 vCPUs, 32 GB RAM — *Default for models up to 14B*
- `g2-standard-24`: 2x `NVIDIA_L4` (48GB total), 24 vCPUs, 96 GB RAM — *Default for 27B–32B models*
- `g2-standard-48`: 4x `NVIDIA_L4` (96GB total), 48 vCPUs, 192 GB RAM — *Default for 70B in FP8 / AWQ*
- `g2-standard-96`: 8x `NVIDIA_L4` (192GB total), 96 vCPUs, 384 GB RAM
### 1. G2 Series (NVIDIA L4) - Rigid Hardware Topology Constraints
Google Cloud enforces strict 1:1 hardware topologies for G2 instances. Using an incorrect CPU:GPU pairing results in an immediate `INVALID_ARGUMENT` API rejection:
- `g2-standard-4`, `g2-standard-8`, `g2-standard-12`, `g2-standard-16`, `g2-standard-32`: Strictly 1x `NVIDIA_L4` (24GB). `g2-standard-8` is the standard default for models ≤14B.
- `g2-standard-24`: Strictly 2x `NVIDIA_L4` (48GB total), 24 vCPUs, 96 GB RAM. (Note: `g2-standard-16` cannot be paired with 2x L4).
- `g2-standard-48`: Strictly 4x `NVIDIA_L4` (96GB total), 48 vCPUs, 192 GB RAM. Minimum safe envelope for ≥27B models (such as Qwen 27B / Gemma 27B) when CUDA Graph memory profiling is active.
- `g2-standard-96`: Strictly 8x `NVIDIA_L4` (192GB total), 96 vCPUs, 384 GB RAM.

### 2. A2 Series (NVIDIA A100)
- `a2-highgpu-1g`: 1x `NVIDIA_TESLA_A100` (40GB), 12 vCPUs, 85 GB RAM
- `a2-highgpu-2g`: 2x `NVIDIA_TESLA_A100` (80GB total), 24 vCPUs, 170 GB RAM
- `a2-ultragpu-1g`: 1x `NVIDIA_A100_80GB` (80GB), 12 vCPUs, 170 GB RAM — *Ideal for 70B in FP8 or 32B BF16*
- `a2-ultragpu-1g`: 1x `NVIDIA_A100_80GB` (80GB), 12 vCPUs, 170 GB RAM — *Contiguous 80 GB HBM2e memory; avoids multi-GPU tensor-parallelism overhead for 27B–32B and 70B FP8 models*
- `a2-ultragpu-2g`: 2x `NVIDIA_A100_80GB` (160GB total), 24 vCPUs, 340 GB RAM
- `a2-ultragpu-4g`: 4x `NVIDIA_A100_80GB` (320GB total), 48 vCPUs, 680 GB RAM

### 3. A3 Series (NVIDIA H100)
- `a3-highgpu-8g`: 8x `NVIDIA_H100_80GB` (640GB total), 208 vCPUs, 1872 GB RAM

---

## Compute Engine (GCE) Machine Configurations (Fallback)

When Vertex AI Endpoints are not viable or explicitly overridden, GCE instances provide raw host control:
- G2 Series: `g2-standard-4`, `g2-standard-8`, `g2-standard-16`, `g2-standard-32`, `g2-standard-96`
- A2 Series: `a2-highgpu-1g` to `8g`, `a2-ultragpu-1g` to `8g`
- A3 Series: `a3-highgpu-8g`
- GCE instances use deeplearning Debian images with standalone Docker containers running `vllm/vllm-openai:latest` and expose port 8000 via VPC firewall rules.

---

## Cloud TPU Accelerator Overview

Cloud TPUs (Tensor Processing Units) offer exceptional price-to-performance and high-bandwidth memory (HBM) for open-weight LLM inference on Vertex AI Prediction Endpoints.

| TPU Generation | Architecture / Chip | HBM / Chip | Memory Bandwidth | Tensor Cores | Inter-Chip Interconnect (ICI) | Hourly Cost (Approx) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TPU v5e (Lite Pod)** | 2x MXU (128x128), VPU | 16 GiB HBM2 | 819 GB/s | BF16 / Int8 | 1.6 Tbps (2D Torus) | ~$1.20 / chip |
| **TPU v6e (Trillium)** | 4x MXU (128x128), VPU | 32 GiB HBM3 | 1,600 GB/s | Native FP8 / BF16 | 3.2 Tbps (2D Torus) | ~$2.20 / chip |
| **TPU v5p** | Frontier MXU (128x128) | 95 GiB HBM3 | 4,800 GB/s | BF16 / FP8 | 4.8 Tbps (3D Torus) | ~$4.20 / chip |

---

## Vertex AI TPU Machine Configurations

Unlike GPUs, Cloud TPU machine types have accelerator chips **hardwired into the machine definition**.

| TPU Generation | Vertex AI `machine_type` | Chip Count | Topology | Total HBM | Host vCPUs | Host RAM | Sizing Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TPU v5e** | `ct5lp-hightpu-1t` | 1 chip | 1x1 | 16 GiB | 24 | 48 GiB | Small models (0.5B - 4B BF16 / 7B FP8) |
| **TPU v5e** | `ct5lp-hightpu-4t` | 4 chips | 2x2 | 64 GiB | 112 | 192 GiB | Mid models (7B - 14B BF16 / 27B-32B FP8) |
| **TPU v5e** | `ct5lp-hightpu-8t` | 8 chips | 2x4 | 128 GiB | 224 | 384 GiB | Large models (14B - 32B BF16 / 70B FP8) |
| **TPU v6e (Trillium)** | `ct6e-standard-1t` | 1 chip | 1x1 | 32 GiB | 24 | 64 GiB | Fast 8B BF16 / 14B FP8 |
| **TPU v6e (Trillium)** | `ct6e-standard-4t` | 4 chips | 2x2 | 128 GiB | 96 | 256 GiB | Dense 27B-32B BF16 / 70B FP8 |
| **TPU v6e (Trillium)** | `ct6e-standard-8t` | 8 chips | 2x4 | 256 GiB | 192 | 512 GiB | Dense 70B BF16 / MoE Flash variants |
| **TPU v5p** | `ct5p-hightpu-4t` | 4 chips | 2x2 | 380 GiB | 120 | 480 GiB | Frontier MoE & ultra-large weights |
| **TPU v5p** | `ct5p-hightpu-8t` | 8 chips | 2x4 | 760 GiB | 240 | 960 GiB | Massive MoE (DeepSeek full-scale) |

### Cardinal Rules for Vertex AI TPU Provisioning

1. **Do NOT pass accelerator parameters**: TPU machine types (`ct5lp-*`, `ct6e-*`, `ct5p-*`) have accelerators hardwired. Passing `--accelerator=type=...,count=...` alongside `ct*` machine types triggers an immediate `INVALID_ARGUMENT` API rejection.
2. **Autoscaling is NOT supported**: TPUs do not support dynamic autoscaling on Vertex AI Endpoints. You must configure `--min-replica-count=1` and `--max-replica-count=1`.
3. **High Shared Memory (`/dev/shm`)**: PyTorch/XLA and Ray multiprocessing require large shared memory. Always configure `--container-shared-memory-size-mb=32768` (32 GiB).
4. **Extended Deployment Timeout**: Initial weight loading and Ahead-Of-Time (AOT) XLA compilation takes 15–30 minutes. Always set `--container-deployment-timeout-seconds=7200` (2 hours).
5. **vLLM-TPU Serving Container**: Use `vllm/vllm-tpu:latest` with environment variables `PJRT_DEVICE=TPU`, `VLLM_TPU_ENABLE_PALLAS=1`, and `RAY_DEDUP_LOGS=0`.
6. **Co-located Cloud Storage Staging**: Stage model weights into a regional GCS bucket in the same region as the endpoint (`us-central1`, `europe-west4`) to avoid Hugging Face rate-limits and container cold-start timeouts.

---

## Cost & Economics: Cloud TPU vs NVIDIA GPU

| Hardware Configuration | Hourly Cost (Approx) | Aggregate Bandwidth | Serving Concurrency | Relative Cost per 1M Tokens |
| :--- | :--- | :--- | :--- | :--- |
| **1x NVIDIA L4 (24GB)** | ~$0.70 / hr | 300 GB/s | 1 - 4 streams | Baseline (1.0x) |
| **4x TPU v5e (`ct5lp-4t`, 64GB)** | ~$4.80 / hr | 3,276 GB/s | 16 - 64 streams | **0.55x (45% Cheaper)** |
| **8x TPU v5e (`ct5lp-8t`, 128GB)** | ~$9.60 / hr | 6,552 GB/s | 32 - 128 streams | **0.50x (50% Cheaper)** |
| **4x TPU v6e (`ct6e-4t`, 128GB)** | ~$8.80 / hr | 6,400 GB/s | 32 - 128 streams | **0.42x (58% Cheaper)** |
| **1x NVIDIA A100 (80GB)** | ~$3.67 / hr | 2,039 GB/s | 8 - 16 streams | 1.10x |
| **8x NVIDIA H100 (80GB)** | ~$80.00 / hr | 26,800 GB/s | 128 - 256 streams | 1.40x |

---

## Quotas and Pre-Flight Verification

Before attempting any deployment, **always verify your project's regional GPU or Cloud TPU quota**. Deploying to an endpoint without sufficient quota will trigger an immediate `RESOURCE_EXHAUSTED` error.

### Automated Quota Verification via fulcrum

`/fulcrum run` automatically audits quota before provisioning. When deploying Cloud TPUs without an explicit `--region`, Fulcrum proactively sweeps all known TPU regions (`europe-west4`, `us-central1`, `asia-southeast1`, `us-east4`), ranks them by available headroom, and automatically routes the deployment to the region with verified capacity.

You can also run pre-flight audits standalone:

```bash
# Proactively sweep all candidate TPU regions for Cloud TPU v6e (Trillium)
.venv/bin/python3 fulcrum/scripts/check_quota.py --is-tpu --tpu-type v6e --count 8 --sweep

# Verify NVIDIA L4 GPU quota on Vertex AI
.venv/bin/python3 fulcrum/scripts/check_quota.py --platform vertex --accelerator nvidia-l4 --count 1 --region us-central1

# Verify Cloud TPU v6e chip quota in a specific region
.venv/bin/python3 fulcrum/scripts/check_quota.py --platform vertex --accelerator v6e --count 4 --region europe-west4 --is-tpu

# Verify NVIDIA A100 GPU quota on Compute Engine (GCE)
.venv/bin/python3 fulcrum/scripts/check_quota.py --platform gce --accelerator nvidia-tesla-a100 --count 1 --region us-central1
```

### Manual Quota Inspection Commands

#### 1. Compute Engine Regional Accelerator Quotas
```bash
# Inspect all regional quotas for target region
gcloud compute regions describe us-central1 --project=YOUR_PROJECT_ID --format="table(quotas.metric,quotas.limit,quotas.usage)"

# Filter specifically for GPU and TPU metrics
gcloud compute regions describe us-central1 --project=YOUR_PROJECT_ID --format="json(quotas)" | jq '.quotas[] | select(.metric | test("NVIDIA|TPU|GPUS"))'
```

#### 2. Vertex AI Custom Model Serving Quotas
Vertex AI maintains dedicated quota metrics under `aiplatform.googleapis.com` queried via Cloud Quotas:
* **NVIDIA L4**: `CustomModelServingNvidiaL4GPUsPerProjectPerRegion` (`custom_model_serving_nvidia_l4_gpus`)
* **NVIDIA A100 (40GB/80GB)**: `CustomModelServingNvidiaA100GPUsPerProjectPerRegion` (`custom_model_serving_nvidia_a100_gpus`)
* **NVIDIA H100**: `CustomModelServingNvidiaH100GPUsPerProjectPerRegion` (`custom_model_serving_nvidia_h100_gpus`)
* **Cloud TPU v6e (Trillium)**: `CustomModelServingV6ETPUPerProjectPerRegion` (`custom_model_serving_tpu_v6e`)
* **Cloud TPU v5e**: `CustomModelServingV5ETPUPerProjectPerRegion` (`custom_model_serving_tpu_v5e`)
* **Cloud TPU v5p**: `CustomModelServingV5pTPUPerProjectPerRegion` (`custom_model_serving_tpu_v5p`)

### Quota Metric Reference Table

| Accelerator | Hardware | GCE Metric | Vertex AI Prediction Metric / Quota ID | Console Filter |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA L4** | 24 GiB VRAM | `NVIDIA_L4_GPUS` | `custom_model_serving_nvidia_l4_gpus` | `NVIDIA_L4` |
| **NVIDIA A100 (40GB)** | 40 GiB VRAM | `NVIDIA_A100_GPUS` | `custom_model_serving_nvidia_a100_gpus` | `NVIDIA_A100` |
| **NVIDIA A100 (80GB)** | 80 GiB VRAM | `NVIDIA_A100_80GB_GPUS` | `custom_model_serving_nvidia_a100_gpus` | `NVIDIA_A100` |
| **NVIDIA H100 (80GB)** | 80 GiB VRAM | `NVIDIA_H100_GPUS` | `custom_model_serving_nvidia_h100_gpus` | `NVIDIA_H100` |
| **Cloud TPU v6e** | 32 GiB HBM / chip | `TPU_V6E_CHIPS` | `custom_model_serving_tpu_v6e` (`CustomModelServingV6ETPUPerProjectPerRegion`) | `custom_model_serving_tpu_v6e` |
| **Cloud TPU v5e** | 16 GiB HBM / chip | `TPU_V5_LITE_PODSLICE_CHIPS` | `custom_model_serving_tpu_v5e` (`CustomModelServingV5ETPUPerProjectPerRegion`) | `custom_model_serving_tpu_v5e` |
| **Cloud TPU v5p** | 95 GiB HBM / chip | `TPU_V5P_CHIPS` | `custom_model_serving_tpu_v5p` (`CustomModelServingV5pTPUPerProjectPerRegion`) | `custom_model_serving_tpu_v5p` |

### How to Request a Quota Increase

1. Open the [Google Cloud Console Quotas & System Limits Page](https://console.cloud.google.com/iam-admin/quotas).
2. Filter by **Service**: `Vertex AI API` (for endpoints) or `Compute Engine API` (for VMs).
3. Filter by **Metric** and target **Region** (e.g. `us-central1`, `europe-west4`).
4. Select the quota line item, click **Edit Quota**, and submit your requested limit increase. Approvals for standard accelerators typically take between a few minutes and 2 business days.

---

## Deployment Best Practices

1. **Verify Quotas Proactively & Rely on Automated Sweeping**:
   Always execute pre-flight quota checks or pass `--strict-quota` to `/fulcrum run` to prevent aborted deployments. For Cloud TPU, Fulcrum automatically sweeps candidate regions (`europe-west4`, `us-central1`, `asia-southeast1`, `us-east4`) and routes to the region with verified capacity.
2. **Preferred Accelerator Regions**:
   - `us-central1` (Iowa): Highest global allocation across L4, A100, H100, and TPU v5e.
   - `europe-west4` (Eemshaven): Primary European accelerator hub; major pool for TPU v6e (Trillium) and v5e.
   - `asia-southeast1` (Singapore) & `us-east4` (Virginia): Strong secondary TPU/GPU hubs.
3. **Shared Memory (`--shm-size`)**:
   - GPU deployments: `--container-shared-memory-size-mb=16384` (16GB).
   - TPU deployments: `--container-shared-memory-size-mb=32768` (32GB).
4. **Vertex AI Model Garden Container Interface Contract (GPUs)**:
   When using `pytorch-vllm-serve:latest`, always explicitly specify `--container-command="python3,-m,vllm.entrypoints.openai.api_server"`, `--container-ports=8080`, `--container-predict-route="/v1/chat/completions"`, and `--container-health-route="/health"`.
5. **CUDA Graph Memory Footprint on ≥20B Models (GPUs)**:
   Ahead-of-Time CUDA Graph workspace takes ~6.0 GiB per GPU on models ≥20B. Either provision 4x L4 (`g2-standard-48`) / 1x A100-80GB (`a2-ultragpu-1g`), or enable `--enforce-eager` to reclaim ~5–6 GiB VRAM per GPU.
6. **Pre-Flight KV-Cache & HBM Memory Budget Sizing**:
   Before submitting Vertex AI model deployments, static pre-flight memory sizing verifies whether available KV cache memory (after loading weights and reserving runtime buffers) can host the requested context length:
   $$\text{Max Safe Context} = \left\lfloor \frac{\text{Total HBM} \times \text{utilization} - \text{Weight Memory} - \text{Runtime Overhead}}{\text{Effective KV Memory Per Token}} \right\rfloor$$
   If requested context exceeds Max Safe Context, `--max-model-len` is automatically clamped (e.g. to 4096 on 4-chip slices for 32B models) and `--gpu-memory-utilization 0.90` is set dynamically to avert initialization crashes.
7. **Decouple Deployments from the 30-Minute Client Socket Timeout**:
   Large models (27B+) on Cloud TPU or multi-GPU instances routinely require 35–45 minutes for container pull, 50+ GiB weight streaming, and Ahead-Of-Time (AOT) JAX/XLA graph compilation. Synchronous `gcloud ai endpoints deploy-model` aborts at 1,800s (30m) with a client socket timeout and does not support `--async`. Always dispatch deployments asynchronously via the Vertex AI REST API (`POST ...:deployModel`), record the Operation ID, and hand off monitoring to an internal progress poller (`gcloud ai operations describe`) with real-time stage updates up to 7,200s (2 hours).


