# vLLM Inference Tuning & Production Runbook

This guide covers advanced tuning parameters for vLLM on GCP to maximise throughput, minimise Time To First Token (TTFT), and optimise GPU memory efficiency.

---

## Key Performance Levers

### 1. Memory Allocation: `--gpu-memory-utilization`
- **Default**: `0.90` (allocates 90% of total VRAM to weights and KV cache blocks).
- **Tuning**:
  - Raise to `0.95` on dedicated inference instances where no other GPU tasks are running.
  - This immediately increases the number of available KV cache blocks by 50% or more, directly expanding maximum concurrency without increasing memory pressure.

### 2. Chunked Prefill: `--enable-chunked-prefill`
- **Problem**: When a new request arrives with a long prompt (e.g. 8k tokens), standard vLLM prefills the entire prompt in one massive iteration, stalling ongoing token decoding for concurrent streams (causing sharp spikes in Time Per Output Token - TPOT).
- **Solution**: `--enable-chunked-prefill` slices prompt prefill into chunks bounded by `--max-num-batched-tokens`. Prefill and decode iterations are co-scheduled smoothly, maintaining low latency for active streaming requests.

### 3. Prefix Caching: `--enable-prefix-caching`
- **Mechanisms**: In conversational chat, multi-turn agentic workflows, or applications with large static system prompts (or few-shot examples), prefixes are identical across requests.
- **Benefit**: Reuses previously calculated KV blocks directly from cache. TTFT drops from hundreds of milliseconds to under 10ms for cached prefixes.

### 4. Continuous Batching Limits: `--max-num-batched-tokens` & `--max-num-seqs`
- **Throughput Optimisation**:
  - Set `--max-num-batched-tokens 4096` (for L4) or `8192` (for A100/H100).
  - Set `--max-num-seqs 128` to `256`.
- **Latency Optimisation (Single-stream / low concurrency)**:
  - Set `--max-num-batched-tokens 1024` or `2048`.
  - Set `--max-num-seqs 16` to `32`.

### 5. Native FP8 Quantisation: `--quantization fp8`
- Supported out-of-the-box on Ada Lovelace (L4) and Hopper (H100).
- Halves memory bandwidth requirements and memory footprint of model weights.
- Doubles generation throughput on memory-bound decode phases.

### 6. Multi-GPU Tensor Parallelism: `--tensor-parallel-size`
- Set equal to the number of physical GPUs (e.g. `--tensor-parallel-size 2` for a 2x L4 instance).
- Ensure GPUs are connected via NVLink (on A100/H100) or high-speed PCIe Gen4 (on G2 instances).

---

## Production Health Checks & Diagnostics

### Endpoints:
- `GET /health`: Returns 200 OK when the model server is live and ready to accept queries.
- `GET /v1/models`: Returns list of currently loaded model identifiers.
- `POST /v1/chat/completions`: Standard OpenAI-compatible conversational endpoint.
- `GET /metrics`: Prometheus-compatible metrics endpoint exposing:
  - `vllm:num_requests_running`
  - `vllm:num_requests_waiting`
  - `vllm:gpu_cache_usage_factor`
  - `vllm:time_to_first_token_seconds`
  - `vllm:time_per_output_token_seconds`

### Memory Pressure Warning Signs:
- If `vllm:num_requests_waiting` grows continuously, incoming traffic exceeds KV cache capacity. Mitigate by increasing `--gpu-memory-utilization` or provisioning an additional replica.
- If GPU OOM occurs during initialisation, reduce `--gpu-memory-utilization` to `0.90` or reduce `--max-model-len`.
### Memory Pressure & CUDA Graph OOM Mitigation:
- **CUDA Graph Workspace Overhead**:
  In vLLM V1 with PyTorch 2.5+, Ahead-of-Time CUDA Graph capture and Torch Dynamo profile batch ranges (1, 2048) during startup (`profile_cudagraph_memory()`), requiring ~5–6 GiB of JIT workspace per GPU. For models ≥20B parameters on constrained hardware (e.g. 27B on 2x L4), this triggers fatal OutOfMemoryError during warm-up.
  - **Solution 1 (Hardware)**: Scale to 4x L4 (`g2-standard-48`) or 1x A100-80GB (`a2-ultragpu-1g`).
  - **Solution 2 (Software Flag)**: Pass `--enforce-eager` to bypass CUDA Graph capture and Torch Dynamo compilation completely, reclaiming ~5–6 GiB of VRAM immediately with negligible throughput penalty.
  - **Solution 3 (Context Cap)**: Cap `--max-model-len 4096` to constrain initial graph buffer sizing.

- **Virtual Address Space Fragmentation**:
  Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the serving container. This enables PyTorch's virtual memory allocator to expand existing memory segments without requiring contiguous physical allocations, preventing OOMs caused by address space fragmentation.

- **Pre-quantised vs Dynamic Runtime Quantisation**:
  Dynamic runtime FP8 (`--quantization fp8` on a BF16 checkpoint) requires `Fp8PerTensorOnlineLinearMethod` activation scale buffers (~1.2 GiB VRAM overhead). Whenever possible, prefer pre-quantised checkpoints (native FP8 safetensors) to eliminate runtime dynamic quantisation overhead.

---

## Benchmarking & Bottleneck Diagnostics

Use `/fulcrum benchmark` (`fulcrum/scripts/benchmark_deployment.py`) to systematically isolate inference bottlenecks across prefill, decode, and queue stages:

| Symptom Pattern | Primary Bottleneck | Remediation Strategy |
| :--- | :--- | :--- |
| **High TTFT, Low TPOT** (e.g. TTFT > 1.5s, TPOT < 15ms) | **Compute/Prefill Bound**: Long prompts stalling attention computation. | Enable `--enable-chunked-prefill` and `--max-num-batched-tokens 2048` to slice prefill iterations; enable `--enable-prefix-caching` if prompts share system headers. |
| **Normal TTFT, High TPOT** (e.g. TTFT < 80ms, TPOT > 35ms) | **Memory Bandwidth Bound**: Autoregressive decode iteration memory transfer limit. | Enable FP8 quantisation (`--quantization fp8` or deploy pre-quantised checkpoint) to double memory bandwidth efficiency; or scale tensor parallelism (`--tensor-parallel-size`). |
| **High ITL Jitter (StdDev > 10ms)** | **Batch Eviction / Context Preemption**: Decode tokens pausing intermittently. | New requests with large prompts are entering without chunking, or KV cache blocks are being evicted. Enable chunked prefill and increase `--gpu-memory-utilization` to `0.95`. |
| **TTFT Inflates Sharply with Concurrency** (e.g. 50ms at c=1, 400ms at c=8) | **KV Cache Saturation / Queue Contention**: Engine running out of physical KV blocks. | Settle on the concurrency knee identified by `/fulcrum benchmark --sweep`. Scale horizontally with multiple replicas or vertically to a GPU with higher total VRAM. |

---

## Cloud TPU & XLA Performance Tuning Runbook

Deploying models on Cloud TPUs requires navigating the **XLA (Accelerated Linear Algebra) Compilation Lifecycle** and PyTorch/XLA PJRT execution runtime.

### 1. The XLA Dynamic Shape Dilemma
Unlike GPUs which dynamically launch CUDA warps for arbitrary token lengths without recompilation, XLA compiles static computational graphs optimised for the physical dimensions of the TPU systolic array (MXU).
- **The Pitfall**: An un-warmed TPU serving container that encounters an unseen sequence length or batch size will trigger an in-flight Just-In-Time (JIT) compilation cycle, stalling the request for 30–60 seconds.
- **The Solution (Static Bucketing)**: Configure sequence length buckets (e.g. 128, 512, 1024, 2048, 4096). Inputs are padded to the nearest bucket, guaranteeing that only a small, finite set of XLA graphs are ever compiled.

### 2. Pallas PagedAttention (`VLLM_TPU_ENABLE_PALLAS=1`)
Pallas is JAX's low-level kernel language for TPUs.
- Standard attention compiles static tensor graphs for sequence lengths, which OOMs or stalls on TPUs when sequence lengths vary.
- Pallas PagedAttention decouples physical KV cache memory management from tensor graph shapes, reading non-contiguous HBM pages directly.
- **Always ensure**: `VLLM_TPU_ENABLE_PALLAS=1` and `PJRT_DEVICE=TPU` are set in the serving container environment.

### 3. Chunked Prefill & Prefix Caching for Extreme Contexts
For models with extreme context windows (e.g. Kimi K3 with 200k+ tokens or DeepSeek MLA):
- Quadratic attention $\mathcal{O}(L^2)$ quickly exhausts HBM on 16 GiB / 32 GiB chips.
- Enable `--enable-chunked-prefill` to slice prompt evaluation into manageable chunks.
- Enable `--enable-prefix-caching` to reuse KV blocks across conversational turns.
- Set a sensible `--max-model-len` (e.g. 32,768 on TPU v5e-8t or v6e-8t).

### 4. Co-located Regional Cloud Storage Staging
- Direct downloads of 15–70 GiB models from Hugging Face at container startup cause rate-limiting (HTTP 429), gateway timeouts, and cold starts exceeding 20 minutes.
- **Best Practice**: Download weights once and rsync them to a regional Cloud Storage bucket co-located in the endpoint region (e.g. `gs://${PROJECT_ID}-tpu-model-staging/models/...` in `us-central1`).
- Mount or stream from the bucket using `gcloud storage rsync` or Cloud Storage FUSE.

### 5. Multi-Chip ICI Tensor Parallelism
- Cloud TPU pods feature optical Inter-Chip Interconnects (ICI) operating at 1.6 Tbps (v5e) to 4.8 Tbps (v5p) in 2D/3D torus topologies.
- Set `--tensor-parallel-size` equal to the physical TPU chip count (1 for `1t`, 4 for `4t`, 8 for `8t`).
- Inter-chip communication overhead is virtually negligible compared to standard PCIe GPU topologies.

### 6. Cloud TPU vLLM Container CLI Differences & Unsupported Flags
- **Strictly Omit `--swap-space`**: In vLLM GPU builds, `--swap-space` configures the host CPU RAM allocated per GPU for offloading KV blocks. In vLLM TPU builds (`vllm/vllm-tpu` using `vllm.entrypoints.openai.api_server`), `--swap-space` is **not supported** and causes Python `argparse` to immediately abort with exit code 2:
  ```
  api_server.py: error: unrecognized arguments: --swap-space=8
  ```
  On Cloud TPUs, KV caching is managed entirely within High Bandwidth Memory (HBM) via Pallas PagedAttention. There is no host CPU RAM swapping mechanism. Never pass `--swap-space` when configuring or deploying on Cloud TPUs.
- **Fixed Machine Topologies (Do Not Pass `--accelerator`)**: Cloud TPU machine types (`ct5lp-*`, `ct6e-*`, `ct5p-*`) have hardwired chip configurations. Supplying `--accelerator=type=...,count=...` triggers an immediate Vertex AI deployment failure (`INVALID_ARGUMENT: Accelerator type must not be specified for TPU machines`).

### 7. Pre-Flight KV-Cache & HBM Memory Budget Calculation
- **The Issue Encountered**:
  When deploying large models (e.g. 32B parameters) on 4-chip TPU slices (`ct6e-standard-4t`, 124.96 GiB HBM) with requested sequence length `8192`, vLLM container warm-up crashes during initialization:
  ```text
  ValueError: To serve at least one request with the model's max seq len (8192), 
  74.77 GiB KV cache is needed, which is larger than the available KV cache memory (54.58 GiB). 
  Based on the available memory, the estimated maximum model length is 5952.
  ```
- **Technical Analysis**:
  After loading ~54–63 GiB of model weights and reserving runtime buffers (~3.88 GiB for XLA/PJRT), only ~45–55 GiB of available KV cache headroom remains. On Cloud TPU with Pallas PagedAttention, block tables and memory allocations scale across the TPU chip mesh with batch buffer reservation (~37.38x stream factor). At default `max_position_embeddings=8192`, serving even a single request exceeds available HBM.
- **Pre-Flight Memory Budget Sizing**:
  Before submitting the deployment, Fulcrum computes:
  $$\text{KV Memory Per Token} = 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes\_per\_elem}$$
  $$\text{Max Safe Context} = \left\lfloor \frac{\text{Total HBM} \times \text{utilization} - \text{Weight Memory} - \text{Runtime Overhead}}{\text{KV Memory Per Token} \times \text{Stream Factor}} \right\rfloor$$
- **Automatic Dynamic Clamping**:
  If the model's requested sequence length exceeds $\text{Max Safe Context}$, Fulcrum automatically clamps `--max-model-len` to the highest safe standard power-of-2 context (e.g. 4096) and sets `--gpu-memory-utilization 0.90` dynamically, preventing hours of failed container warmup cycles.

### 8. Decoupling Vertex Deployment from the 30-Minute Client Timeout
- **The Issue Encountered**:
  Large models (27B–70B+) on Cloud TPU or multi-GPU topologies require image container pull, streaming 50+ GiB of weights from Hugging Face or Cloud Storage, and Ahead-Of-Time (AOT) JAX/XLA graph compilation across multiple sequence buckets (`128`, `256`, `512`, `1024`, `2048`, `4096`).
  Total container warmup routinely takes 35 to 45 minutes. At the 30-minute mark, synchronous `gcloud ai endpoints deploy-model` terminates with a client socket timeout:
  ```text
  Operation [projects/.../locations/.../operations/1810106235958067200] has not finished in 1800 seconds
  ```
  An uninformed agent or user might assume this was an infrastructure failure and cancel the operation, even though the remote operation was progressing smoothly. Furthermore, `gcloud ai endpoints deploy-model` does **not** support an `--async` flag (`unrecognized arguments: --async`).
- **Asynchronous REST Dispatch**:
  To decouple client orchestration from the 30-minute client socket timeout, Fulcrum dispatches deployments asynchronously via the Vertex AI REST API:
  ```bash
  curl -s -X POST \
      -H "Authorization: Bearer $(gcloud auth print-access-token)" \
      -H "Content-Type: application/json" \
      "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT}/locations/${REGION}/endpoints/${ENDPOINT_ID}:deployModel" \
      -d '{
        "deployedModel": {
          "model": "projects/'"${PROJECT}"'/locations/'"${REGION}"'/models/'"${MODEL_ID}"'",
          "displayName": "'"${ENDPOINT_NAME}"'-deployed",
          "dedicatedResources": {
            "machineSpec": {"machineType": "'"${MACHINE_TYPE}"'"},
            "minReplicaCount": 1,
            "maxReplicaCount": 1
          }
        },
        "trafficSplit": {"0": 100}
      }'
  ```
  This returns immediately (<1 second) with the Long-Running Operation (`projects/.../operations/{op_id}`).
- **Dedicated Operation Polling & Timeout Recovery**:
  Fulcrum immediately records the `operation_id` in its state database and initiates a progress poller via:
  ```bash
  gcloud ai operations describe ${OP_ID} --project=${PROJECT} --region=${REGION} --format=json
  ```
  Fulcrum monitors the deployment with real-time stage updates up to 7,200 seconds (2 hours). If standard `gcloud` CLI was used and hit the 1,800s timeout, Fulcrum automatically intercepts the Operation ID from `stderr` and hands off to the operation poller without failing the workflow.

### 9. Handling Transient Datacentre Allocation Stock-Outs (Code 14 UNAVAILABLE)
- **The Issue Encountered**:
  During model deployment on high-demand accelerator topologies (such as Cloud TPU `ct6e-standard-4t` or NVIDIA H100/A100 instances), the deployment operation may fail during node allocation with:
  ```text
  code: 14: Machine type temporarily unavailable, please deploy with a different machine type or retry. Machine type: "ct6e-standard-4t"
  ```
- **Root Cause & Technical Analysis**:
  - **Differentiating Quota vs Physical Capacity**:
    - **Code 14 (`UNAVAILABLE`)**: This is **NOT** a quota exhaustion (Code 8 `RESOURCE_EXHAUSTED`) or an invalid configuration (Code 3 `INVALID_ARGUMENT`). Project quota is completely **intact**, and the machine type specification is completely **valid**.
    - Code 14 represents transient physical node contention in the datacentre. Physical TPU or GPU host racks in that specific zone are temporarily occupied by other multi-tenant training, fine-tuning, or inference jobs.
    - Cloud datacentre node contention is ephemeral: as other batch workloads conclude, nodes are released back into the unallocated pool (often within 5 to 15 minutes).
- **Automated Diagnosis & Actionable Guidance**:
  Fulcrum explicitly intercepts Code 14 across async operation polling and CLI fallback:
  - Reassures the operator that quota is intact and configuration is valid.
  - Updates the deployment status to `STOCKOUT` in the local state database.
  - Recommends immediate operational remedies: wait 5–15 minutes and retry, or route to an alternate region with verified capacity.
- **Automated Exponential Backoff Retry**:
  Fulcrum provides native automated retry capabilities via `--retry-transient`:
  ```bash
  python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen2.5-14B-Instruct \
      --tpu-type v6e --tpu-chips 4 --region europe-west4 \
      --retry-transient --max-retries 2 --retry-delay 300
  ```
  When `--retry-transient` is enabled, Fulcrum catches Code 14, logs an explanation banner, and executes exponential backoff retries (`delay = retry_delay * (1.5 ** attempt)`). As demonstrated in production sessions, retrying 15 minutes later succeeded immediately once nodes became free.

---

## 10. Dedicated Cloud Logging & Live Container Telemetry

Vertex AI exposes minimal startup feedback in the Google Cloud Console while a container is booting, streaming weights, or compiling XLA graphs. Fulcrum provides a native logging engine embedded directly in `/fulcrum status`.

### Architecture:
- **Numeric Endpoint ID Resolution**: Cloud Logging labels container output under `resource.type="aiplatform.googleapis.com/Endpoint"` and `resource.labels.endpoint_id="<NUMERIC_ID>"`. Fulcrum automatically translates user-friendly endpoint display names into the canonical numeric ID required by the Cloud Logging API.
- **Semantic Component Tagging**: Rather than dumping unparsed JSON blobs, Fulcrum cleanses log entries and categorises them into high-visibility component tags:
  - `[Weight Streaming]`: Progress of checkpoint downloading (safetensors shards, GCS stage transfers).
  - `[JAX/XLA Compilation]`: Ahead-Of-Time TPU compilation progress across sequence buckets (`128`, `256`, `512`, `1024`, `2048`, `4096`).
  - `[vLLM EngineCore]`: Core model runner initialisation, tensor parallel worker rank assignment, and KV-cache allocation.
  - `[vLLM APIServer]`: HTTP routing, FastAPI/Uvicorn listener status, and OpenAI-compatible `/v1/chat/completions` readiness.
  - `[Error]` / `[Warning]`: Highlighted warning and error messages.

### Command Usage:
```bash
# Fetch and format recent 50 container log entries:
.venv/bin/python3 fulcrum/scripts/status_deployment.py --id <resource-name> --logs

# Stream and follow live container startup logs in real time:
.venv/bin/python3 fulcrum/scripts/status_deployment.py --id <resource-name> --logs --follow

# Inspect raw JSON log payloads for low-level debugging:
.venv/bin/python3 fulcrum/scripts/status_deployment.py --id <resource-name> --logs --raw
```
