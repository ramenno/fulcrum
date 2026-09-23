#!/usr/bin/env python3
"""
Verification test suite for all /fulcrum skill scripts.
Ensures find_model, launch_gcp, status_deployment, optimise_deployment,
delete_deployment, benchmark_deployment, chat_web, check_quota, and db state operations
pass all checks across Vertex AI Endpoints (primary default) and Compute Engine (fallback).
"""

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def run(cmd: str, env_vars: dict = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"FAILED: {cmd}\nStderr: {proc.stderr}")
        sys.exit(proc.returncode)
    return proc


def test_db_operations() -> None:
    print("Testing db.py state operations...")
    test_db = "/tmp/fulcrum_unit_test_db.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    # Test programmatic CRUD via python
    code = f"""
import os
os.environ['FULCRUM_DB_PATH'] = '{test_db}'
from fulcrum.scripts import db

# Clear and verify empty
db.clear_deployments()
assert len(db.load_deployments()) == 0

# Add Vertex record
rec = {{
    'id': 'fulcrum-test-model',
    'endpoint_name': 'fulcrum-endpoint-abc',
    'endpoint_id': '1234567890',
    'platform': 'vertex',
    'model': 'google/gemma-3-12b-it',
    'project': 'test-proj',
    'region': 'us-central1',
    'machine_type': 'g2-standard-8',
    'accelerator': 'NVIDIA_L4',
    'gpu_count': 1,
    'status': 'RECORDED',
}}
db.add_deployment(rec)
assert len(db.load_deployments()) == 1

# Retrieve record by id, endpoint_name, and endpoint_id
item = db.get_deployment('fulcrum-test-model')
assert item is not None
assert item['model'] == 'google/gemma-3-12b-it'
assert item['platform'] == 'vertex'
assert db.get_deployment('fulcrum-endpoint-abc') is not None
assert db.get_deployment('1234567890') is not None

# Update record
db.update_deployment('fulcrum-test-model', {{'status': 'ACTIVE'}})
item2 = db.get_deployment('fulcrum-test-model')
assert item2['status'] == 'ACTIVE'

# Test GCP project auto-detection from gcloud
detected = db.detect_gcp_project()
assert detected is not None
assert len(detected) > 0
resolved = db.resolve_gcp_project()
assert resolved == detected
assert db.resolve_gcp_project('explicit-test-proj') == 'explicit-test-proj'

# Remove record
removed = db.remove_deployment('fulcrum-test-model')
assert removed is not None
assert len(db.load_deployments()) == 0
"""
    run(f'.venv/bin/python3 -c "{code}"')
    if os.path.exists(test_db):
        os.remove(test_db)
    print("db.py: PASSED")


def test_find_model() -> None:
    print("Testing find_model.py...")
    run(".venv/bin/python3 fulcrum/scripts/find_model.py --help")

    p = run(".venv/bin/python3 fulcrum/scripts/find_model.py --task coding --license permissive --format json")
    data = json.loads(p.stdout)
    assert len(data) > 0, "Expected at least one coding model matching criteria"
    for item in data:
        assert "coding" in item["metadata"]["primary_tasks"]
        assert "apache" in item["metadata"]["license"].lower() or "mit" in item["metadata"]["license"].lower()

    p = run(".venv/bin/python3 fulcrum/scripts/find_model.py --query gemma-3-12b --format json")
    data = json.loads(p.stdout)
    calc = data[0]["hardware_calc"]
    assert calc["total_vram_needed_gb"] > 0
    assert "recommended_gcp" in calc
    assert "recommended_tpu" in calc
    assert calc["recommended_tpu"]["machine"].startswith("ct")

    p_tpu_cli = run(".venv/bin/python3 fulcrum/scripts/find_model.py --query qwen --accelerator tpu")
    assert "RECOMMENDED CLOUD TPU" in p_tpu_cli.stdout

    p_qwen = run(".venv/bin/python3 fulcrum/scripts/find_model.py --query qwen2.5-32b --format json")
    data_qwen = json.loads(p_qwen.stdout)
    calc_qwen = data_qwen[0]["hardware_calc"]
    rec_gpu = calc_qwen["recommended_gcp"]
    if rec_gpu["gpu"] == "nvidia-l4":
        assert rec_gpu["count"] >= 4, ">=20B models on L4 must require at least 4 GPUs to prevent CUDA Graph OOM"
    rec_tpu = calc_qwen["recommended_tpu"]
    assert rec_tpu["chips"] >= 4, ">=30B models on TPU require at least 4 chips"
    print("find_model.py: PASSED")


def test_launch_gcp() -> None:
    print("Testing launch_gcp.py...")
    test_db = "/tmp/fulcrum_launch_test_db.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --help")

    # 1. Test Default Priority: Vertex AI Endpoint (Dry-Run)
    p_vertex = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model google/gemma-3-12b-it --dry-run")
    assert "Vertex AI Endpoint (Managed Prediction Service)" in p_vertex.stdout
    assert "gcloud ai endpoints create" in p_vertex.stdout
    assert "gcloud ai models upload" in p_vertex.stdout
    assert "gcloud ai endpoints deploy-model" in p_vertex.stdout

    # 2. Test JSON output for Vertex AI
    p_json = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model meta-llama/Llama-3.1-8B-Instruct --format json")
    data = json.loads(p_json.stdout)
    assert data["platform"] == "vertex"
    assert "create_endpoint" in data["commands"]
    assert "upload_model" in data["commands"]
    assert "deploy_model" in data["commands"]

    # 3. Test Terraform format for Vertex AI
    p_tf = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model google/gemma-3-12b-it --format terraform")
    assert 'resource "google_vertex_ai_endpoint"' in p_tf.stdout
    assert 'resource "google_vertex_ai_model"' in p_tf.stdout
    assert 'resource "google_vertex_ai_endpoint_deployed_model"' in p_tf.stdout

    # 4. Test Compute Engine fallback explicitly (Dry-Run)
    p_gce = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model google/gemma-3-12b-it --platform gce --dry-run")
    assert "Compute Engine (GCE GPU Virtual Machine)" in p_gce.stdout
    assert "gcloud compute instances create" in p_gce.stdout
    assert "fulcrum-allow-vllm" in p_gce.stdout

    # 5. Test multi-node fallback to GKE
    p_gke = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model deepseek-ai/DeepSeek-R1 --platform gke")
    assert "Google Kubernetes Engine (GKE Multi-Node Cluster)" in p_gke.stdout
    assert "kind: Deployment" in p_gke.stdout

    # 6. Test --record into database in dry-run mode
    run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model google/gemma-3-12b-it --dry-run --record", env_vars={"FULCRUM_DB_PATH": test_db})
    with open(test_db, "r") as f:
        recorded = json.load(f)
    assert len(recorded) == 1
    assert recorded[0]["model"] == "google/gemma-3-12b-it"
    assert recorded[0]["platform"] == "vertex"
    assert recorded[0]["status"] == "RECORDED"

    # 7. Test Layer 1: Rigid G2 Topology resolution (g2-standard-24 for 2x L4, g2-standard-48 for 4x L4)
    p_2xl4 = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model meta-llama/Llama-3.1-8B-Instruct --gpu-type nvidia-l4 --gpu-count 2 --format json")
    data_2xl4 = json.loads(p_2xl4.stdout)
    assert data_2xl4["machine_type"] == "g2-standard-24", f"Expected g2-standard-24, got {data_2xl4['machine_type']}"

    # 8. Test Layer 2: Vertex container command contract, rawPredict route, and uppercase accelerator type
    p_contract = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model google/gemma-3-12b-it --format json")
    data_contract = json.loads(p_contract.stdout)
    upload_cmd = data_contract["commands"]["upload_model"]
    deploy_cmd = data_contract["commands"]["deploy_model"]
    sample_cmd = data_contract["commands"]["sample_request"]
    assert '--container-command="python3,-m,vllm.entrypoints.openai.api_server"' in upload_cmd
    assert '--container-predict-route="/v1/chat/completions"' in upload_cmd
    assert '--container-health-route="/health"' in upload_cmd
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in upload_cmd
    assert "--accelerator=type=NVIDIA_L4,count=1" in deploy_cmd
    assert ":rawPredict" in sample_cmd

    # 9. Test Layer 3: Automatic --enforce-eager memory guardrail for >=20B models on <=2x L4
    p_eager = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen3.8-27B --gpu-type nvidia-l4 --gpu-count 2 --format json")
    data_eager = json.loads(p_eager.stdout)
    assert "--enforce-eager" in data_eager["commands"]["upload_model"]

    # 10. Test automatic sizing of 27B model on G2 defaults to 4x L4 (g2-standard-48)
    p_27b = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen3.8-27B --format json")
    data_27b = json.loads(p_27b.stdout)
    assert data_27b["gpu_count"] == 4
    assert data_27b["machine_type"] == "g2-standard-48"

    # 11. Test resolve_machine_type and is_tpu_machine helpers directly
    code_resolve = """
from fulcrum.scripts.launch_gcp import resolve_machine_type, is_tpu_machine
assert resolve_machine_type('nvidia-l4', 1) == 'g2-standard-8'
assert resolve_machine_type('nvidia-l4', 2) == 'g2-standard-24'
assert resolve_machine_type('nvidia-l4', 4) == 'g2-standard-48'
assert resolve_machine_type('nvidia-l4', 8) == 'g2-standard-96'
assert resolve_machine_type('nvidia-a100-80gb', 1) == 'a2-ultragpu-1g'
assert resolve_machine_type('v5e', 4) == 'ct5lp-hightpu-4t'
assert resolve_machine_type('v6e', 4) == 'ct6e-standard-4t'
assert resolve_machine_type('v5p', 4) == 'ct5p-hightpu-4t'
assert is_tpu_machine('ct5lp-hightpu-4t') is True
assert is_tpu_machine('ct6e-standard-4t') is True
assert is_tpu_machine('g2-standard-48') is False
"""
    run(f'.venv/bin/python3 -c "{code_resolve}"')

    # 12. Test Cloud TPU Vertex deployment specification
    p_tpu = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen2.5-14B-Instruct --tpu-type v6e --tpu-chips 4 --format json")
    data_tpu = json.loads(p_tpu.stdout)
    assert data_tpu["machine_type"] == "ct6e-standard-4t"
    assert data_tpu["accelerator"] == "4x Cloud TPU V6E chips"
    tpu_upload = data_tpu["commands"]["upload_model"]
    tpu_deploy = data_tpu["commands"]["deploy_model"]
    assert "vllm/vllm-tpu:latest" in tpu_upload
    assert "PJRT_DEVICE=TPU" in tpu_upload
    assert "VLLM_TPU_ENABLE_PALLAS=1" in tpu_upload
    assert "--container-shared-memory-size-mb=32768" in tpu_upload
    assert "--container-deployment-timeout-seconds=7200" in tpu_upload
    assert "--swap-space" not in tpu_upload
    assert "--accelerator=" not in tpu_deploy
    assert "--accelerator " not in tpu_deploy

    # 13. Test Cloud TPU weight staging via GCS
    p_tpu_gcs = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen2.5-14B-Instruct --tpu-type v6e --tpu-chips 4 --gcs-model-path gs://test-bucket/models/qwen --format json")
    data_tpu_gcs = json.loads(p_tpu_gcs.stdout)
    assert "GCS_MODEL_PATH=gs://test-bucket/models/qwen" in data_tpu_gcs["commands"]["upload_model"]
    assert "--model=/tmp/model" in data_tpu_gcs["commands"]["upload_model"]

    # 14. Test Cloud TPU Terraform generation
    p_tpu_tf = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen2.5-14B-Instruct --tpu-type v6e --tpu-chips 4 --format terraform")
    assert 'machine_type      = "ct6e-standard-4t"' in p_tpu_tf.stdout
    assert "accelerator_type" not in p_tpu_tf.stdout
    assert "accelerator_count" not in p_tpu_tf.stdout
    assert 'image_uri     = "vllm/vllm-tpu:latest"' in p_tpu_tf.stdout

    if os.path.exists(test_db):
        os.remove(test_db)
    print("launch_gcp.py: PASSED")


def test_status_deployment() -> None:
    print("Testing status_deployment.py...")
    test_db = "/tmp/fulcrum_status_test_db.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    run(".venv/bin/python3 fulcrum/scripts/status_deployment.py --help")

    # Populate test database with both Vertex and GCE deployments
    dummy_deployments = [
        {
            "id": "fulcrum-gemma-3-12b-it",
            "endpoint_name": "fulcrum-gemma-3-12b-it",
            "platform": "vertex",
            "model": "google/gemma-3-12b-it",
            "project": "test-proj",
            "region": "us-central1",
            "machine_type": "g2-standard-8",
            "accelerator": "NVIDIA_L4",
            "gpu_count": 1,
            "status": "RECORDED",
            "created_at": "2026-09-09T00:00:00Z",
        },
        {
            "id": "fulcrum-llama-3-1-8b",
            "instance_name": "fulcrum-llama-3-1-8b",
            "platform": "gce",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "project": "test-proj",
            "zone": "us-central1-a",
            "machine_type": "g2-standard-8",
            "gpu_type": "nvidia-l4",
            "gpu_count": 1,
            "port": 8000,
            "status": "RECORDED",
            "created_at": "2026-09-09T00:00:00Z",
        },
    ]
    with open(test_db, "w") as f:
        json.dump(dummy_deployments, f)

    # Test table output
    p_table = run(".venv/bin/python3 fulcrum/scripts/status_deployment.py", env_vars={"FULCRUM_DB_PATH": test_db})
    assert "VERTEX" in p_table.stdout
    assert "GCE" in p_table.stdout
    assert "fulcrum-gemma-3-12b-it" in p_table.stdout
    assert "fulcrum-llama-3-1-8b" in p_table.stdout

    # Test JSON output
    p_json = run(".venv/bin/python3 fulcrum/scripts/status_deployment.py --format json", env_vars={"FULCRUM_DB_PATH": test_db})
    data = json.loads(p_json.stdout)
    assert len(data) == 2
    assert data[0]["platform"] == "vertex"
    assert data[1]["platform"] == "gce"

    # Test verbose diagnostics for Vertex
    p_verb = run(".venv/bin/python3 fulcrum/scripts/status_deployment.py --id fulcrum-gemma-3-12b-it --verbose", env_vars={"FULCRUM_DB_PATH": test_db})
    assert "Vertex Predict URI" in p_verb.stdout

    if os.path.exists(test_db):
        os.remove(test_db)
    print("status_deployment.py: PASSED")


def test_optimise_deployment() -> None:
    print("Testing optimise_deployment.py...")
    run(".venv/bin/python3 fulcrum/scripts/optimise_deployment.py --help")

    p = run(".venv/bin/python3 fulcrum/scripts/optimise_deployment.py advise --model google/gemma-3-12b-it --gpu-type nvidia-l4 --workload throughput --format json")
    data = json.loads(p.stdout)
    assert "--quantization" in data["flags"]
    assert data["flags"]["--quantization"] == "fp8"
    assert data["flags"]["--enable-chunked-prefill"] is True
    assert data["flags"]["--enable-prefix-caching"] is True

    # Test memory-constrained advice for 27B on 2x L4
    p_tight = run(".venv/bin/python3 fulcrum/scripts/optimise_deployment.py advise --model Qwen/Qwen3.8-27B --gpu-type nvidia-l4 --gpu-count 2 --format json")
    data_tight = json.loads(p_tight.stdout)
    assert data_tight["flags"].get("--enforce-eager") is True
    assert any("PYTORCH_CUDA_ALLOC_CONF" in r for r in data_tight["rationales"])
    assert any("pre-quantised" in r for r in data_tight["rationales"])

    # Test Cloud TPU tuning advice
    p_tpu_opt = run(".venv/bin/python3 fulcrum/scripts/optimise_deployment.py advise --model Qwen/Qwen2.5-14B-Instruct --gpu-type ct6e-standard-4t --gpu-count 4 --format json")
    data_tpu_opt = json.loads(p_tpu_opt.stdout)
    assert data_tpu_opt["flags"]["--gpu-memory-utilization"] == 0.85
    assert "--swap-space" not in data_tpu_opt["flags"]
    assert data_tpu_opt["flags"]["--block-size"] == 16
    assert any("Omit --swap-space" in r for r in data_tpu_opt["rationales"])
    assert any("Pallas" in r for r in data_tpu_opt["rationales"])
    assert any("Static Sequence Length Bucketing" in r for r in data_tpu_opt["rationales"])
    assert any("Do NOT pass --accelerator" in r for r in data_tpu_opt["rationales"])
    print("optimise_deployment.py: PASSED")


def test_delete_deployment() -> None:
    print("Testing delete_deployment.py...")
    test_db = "/tmp/fulcrum_delete_test_db.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    run(".venv/bin/python3 fulcrum/scripts/delete_deployment.py --help")

    # Set up test deployments: one Vertex endpoint and two GCE instances sharing port 8000
    dummy_deployments = [
        {
            "id": "fulcrum-vertex-ep",
            "endpoint_name": "fulcrum-vertex-ep",
            "platform": "vertex",
            "model": "google/gemma-3-12b-it",
            "project": "test-proj",
            "region": "us-central1",
            "status": "RECORDED",
        },
        {
            "id": "fulcrum-gce-1",
            "instance_name": "fulcrum-gce-1",
            "platform": "gce",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "project": "test-proj",
            "zone": "us-central1-a",
            "port": 8000,
            "firewall_rule": "fulcrum-allow-vllm-8000",
            "status": "RECORDED",
        },
        {
            "id": "fulcrum-gce-2",
            "instance_name": "fulcrum-gce-2",
            "platform": "gce",
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "project": "test-proj",
            "zone": "us-central1-a",
            "port": 8000,
            "firewall_rule": "fulcrum-allow-vllm-8000",
            "status": "RECORDED",
        },
    ]
    with open(test_db, "w") as f:
        json.dump(dummy_deployments, f)

    # 1. Test Vertex AI deletion plan: should undeploy model, delete endpoint, and delete model
    p_vertex = run(".venv/bin/python3 fulcrum/scripts/delete_deployment.py --id fulcrum-vertex-ep --format json", env_vars={"FULCRUM_DB_PATH": test_db})
    vertex_data = json.loads(p_vertex.stdout)
    assert vertex_data["target_count"] == 1
    op_types = [op["type"] for op in vertex_data["operations"]]
    assert "vertex_undeploy" in op_types
    assert "vertex_endpoint" in op_types
    assert "vertex_model" in op_types

    # 2. Test single GCE instance dry-run: firewall rule should NOT be scheduled because fulcrum-gce-2 still needs it
    p_single = run(".venv/bin/python3 fulcrum/scripts/delete_deployment.py --id fulcrum-gce-1 --format json", env_vars={"FULCRUM_DB_PATH": test_db})
    single_data = json.loads(p_single.stdout)
    assert single_data["target_count"] == 1
    assert len(single_data["operations"]) == 1
    assert single_data["operations"][0]["type"] == "instance"

    # 3. Test --all dry-run: both Vertex operations and GCE instance + firewall teardown
    p_all = run(".venv/bin/python3 fulcrum/scripts/delete_deployment.py --all --format json", env_vars={"FULCRUM_DB_PATH": test_db})
    all_data = json.loads(p_all.stdout)
    assert all_data["target_count"] == 3
    all_op_types = [op["type"] for op in all_data["operations"]]
    assert "vertex_endpoint" in all_op_types
    assert "instance" in all_op_types
    assert "firewall" in all_op_types

    if os.path.exists(test_db):
        os.remove(test_db)
    print("delete_deployment.py: PASSED")


def test_benchmark_deployment() -> None:
    print("Testing benchmark_deployment.py...")
    # 1. Help CLI validation
    run(".venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --help")

    # 2. Mock execution in JSON format: verify metrics structure and percentiles
    p_json = run(
        ".venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --model google/gemma-3-12b-it --mock --concurrency 2 --requests 4 --format json"
    )
    data = json.loads(p_json.stdout)
    assert data["model"] == "google/gemma-3-12b-it"
    assert data["workload_profile"] == "chat"
    res = data["results"]
    assert res["success"] is True
    assert res["concurrency"] == 2
    assert res["successful_requests"] == 4
    assert res["generation_throughput_tokens_per_sec"] > 0
    assert res["prompt_throughput_tokens_per_sec"] > 0
    assert res["request_throughput_rps"] > 0
    assert res["itl_jitter_ms"] >= 0.0

    # Percentiles validation
    for metric in ["ttft_ms", "tpot_ms", "e2e_latency_ms"]:
        assert metric in res
        for pct in ["min", "mean", "p50", "p90", "p95", "p99", "max"]:
            assert pct in res[metric]
            assert isinstance(res[metric][pct], (int, float))

    # 3. Workload profiles: test rag and code profiles
    p_rag = run(
        ".venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --model google/gemma-3-12b-it --workload rag --mock --format json"
    )
    data_rag = json.loads(p_rag.stdout)
    assert data_rag["workload_profile"] == "rag"
    assert data_rag["max_tokens"] == 80

    p_code = run(
        ".venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --model google/gemma-3-12b-it --workload code --mock --format json"
    )
    data_code = json.loads(p_code.stdout)
    assert data_code["workload_profile"] == "code"
    assert data_code["max_tokens"] == 450

    # 4. Concurrency sweep testing
    p_sweep = run(
        ".venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --model google/gemma-3-12b-it --sweep --mock --format json"
    )
    data_sweep = json.loads(p_sweep.stdout)
    assert data_sweep["sweep_mode"] is True
    assert isinstance(data_sweep["results"], list)
    assert len(data_sweep["results"]) >= 3
    sat = data_sweep["saturation_analysis"]
    assert sat is not None
    assert "optimal_concurrency" in sat
    assert "peak_throughput_tps" in sat
    assert "throughput_scaling_factor" in sat

    # 5. Markdown report generation test
    report_path = "/tmp/fulcrum_test_report.md"
    if os.path.exists(report_path):
        os.remove(report_path)
    run(
        f".venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --model google/gemma-3-12b-it --mock --report-file {report_path}"
    )
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "# Benchmark Report" in md_text
    assert "Generation Throughput" in md_text
    assert "TTFT (Prefill)" in md_text
    if os.path.exists(report_path):
        os.remove(report_path)

    # 6. Database record integration and fulcrum status reflection
    test_db = "/tmp/fulcrum_bench_unit_test_db.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    dummy = [
        {
            "id": "fulcrum-test-bench",
            "endpoint_name": "fulcrum-test-bench",
            "platform": "vertex",
            "model": "google/gemma-3-12b-it",
            "project": "test-proj",
            "region": "us-central1",
            "status": "RECORDED",
        }
    ]
    with open(test_db, "w", encoding="utf-8") as f:
        json.dump(dummy, f)

    run(
        ".venv/bin/python3 fulcrum/scripts/benchmark_deployment.py --id fulcrum-test-bench --mock --format json",
        env_vars={"FULCRUM_DB_PATH": test_db},
    )

    with open(test_db, "r", encoding="utf-8") as f:
        updated_db = json.load(f)
    dep_record = updated_db[0]
    assert "last_benchmark" in dep_record
    assert dep_record["last_benchmark"]["throughput_tps"] > 0
    assert "ttft_p50_ms" in dep_record["last_benchmark"]

    # Verify status_deployment displays benchmark info
    p_stat = run(".venv/bin/python3 fulcrum/scripts/status_deployment.py --format json", env_vars={"FULCRUM_DB_PATH": test_db})
    stat_json = json.loads(p_stat.stdout)
    assert len(stat_json) == 1
    assert "last_benchmark" in stat_json[0]

    # Verify status table output has BENCHMARK column
    p_stat_tbl = run(".venv/bin/python3 fulcrum/scripts/status_deployment.py", env_vars={"FULCRUM_DB_PATH": test_db})
    assert "BENCHMARK" in p_stat_tbl.stdout

    if os.path.exists(test_db):
        os.remove(test_db)

    print("benchmark_deployment.py: PASSED")


def test_chat_web() -> None:
    """Automated verification of the fulcrum chat web server and SSE telemetry stream."""
    try:
        from fulcrum.scripts.chat_web import TokenManager, resolve_deployment_target, ChatRequestHandler
    except ModuleNotFoundError:
        from chat_web import TokenManager, resolve_deployment_target, ChatRequestHandler

    print("Testing chat_web.py...")

    # 1. CLI help verification
    p_help = run(".venv/bin/python3 fulcrum/scripts/chat_web.py --help")
    assert "--port" in p_help.stdout
    assert "--mock" in p_help.stdout
    assert "--no-browser" in p_help.stdout
    assert "--id" in p_help.stdout

    # 2. TokenManager unit tests
    tm = TokenManager(ttl_seconds=60)
    assert tm._token is None
    os.environ["GCP_TOKEN"] = "unit-test-token"
    assert tm.get_token() == "unit-test-token"
    del os.environ["GCP_TOKEN"]

    # 3. Endpoint target resolution tests
    url, m, _, _ = resolve_deployment_target(
        endpoint="https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/endpoints/123",
        model="Qwen/Qwen3.8-27B",
    )
    assert url.endswith(":rawPredict")
    assert m == "Qwen/Qwen3.8-27B"

    url_gce, _, _, _ = resolve_deployment_target(endpoint="http://10.0.0.1:8000", model="test")
    assert url_gce.endswith("/chat/completions")

    # 4. In-memory MockSocket client harness for sandbox compatibility
    class MockSocket:
        def __init__(self, data: bytes):
            self._r = io.BytesIO(data)
            self._w = io.BytesIO()

        def makefile(self, mode, *args, **kwargs):
            if "r" in mode:
                return self._r
            return self._w

        def sendall(self, b: bytes) -> None:
            self._w.write(b)

        def flush(self) -> None:
            pass

    class DummyServer:
        server_name = "localhost"
        server_port = 8080

    test_db = "/tmp/fulcrum_chat_unit_test_db.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    dummy_deployments = [
        {
            "id": "fulcrum-chat-unit-test",
            "endpoint_name": "fulcrum-chat-unit-test",
            "platform": "vertex",
            "model": "google/gemma-3-12b-it",
            "project": "test-project-123",
            "region": "us-central1",
            "status": "RECORDED",
        }
    ]
    with open(test_db, "w", encoding="utf-8") as f:
        json.dump(dummy_deployments, f)

    os.environ["FULCRUM_DB_PATH"] = test_db
    ChatRequestHandler.server_context = {
        "mock": True,
        "deployment_id": "fulcrum-chat-unit-test",
        "model": "google/gemma-3-12b-it",
    }

    try:
        # Test GET / (HTML Front-End)
        sock_get = MockSocket(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        ChatRequestHandler(sock_get, ("127.0.0.1", 1234), DummyServer())
        resp_html = sock_get._w.getvalue()
        assert b"HTTP/1.0 200 OK" in resp_html
        assert b"/fulcrum chat" in resp_html
        assert b"stats-hud" in resp_html
        assert b"deployment-select" in resp_html

        # Test GET /api/deployments
        sock_deps = MockSocket(b"GET /api/deployments HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        ChatRequestHandler(sock_deps, ("127.0.0.1", 1234), DummyServer())
        resp_deps = sock_deps._w.getvalue()
        assert b"HTTP/1.0 200 OK" in resp_deps
        assert b"fulcrum-chat-unit-test" in resp_deps

        # Test GET /api/status
        sock_stat = MockSocket(b"GET /api/status HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        ChatRequestHandler(sock_stat, ("127.0.0.1", 1234), DummyServer())
        resp_stat = sock_stat._w.getvalue()
        assert b"HTTP/1.0 200 OK" in resp_stat
        assert b'"mock_mode": true' in resp_stat

        # Test POST /api/chat/stream (SSE token streaming + live telemetry)
        chat_body = json.dumps({
            "messages": [{"role": "user", "content": "Explain KV cache optimizations."}],
            "deployment_id": "fulcrum-chat-unit-test",
            "max_tokens": 32,
        }).encode("utf-8")
        headers = (
            b"POST /api/chat/stream HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(chat_body)).encode("ascii") + b"\r\n"
            b"Connection: close\r\n\r\n"
        )
        sock_stream = MockSocket(headers + chat_body)
        ChatRequestHandler(sock_stream, ("127.0.0.1", 1234), DummyServer())
        raw_stream = sock_stream._w.getvalue()
        assert b"HTTP/1.0 200 OK" in raw_stream
        assert b"text/event-stream" in raw_stream

        # Parse received SSE events
        tokens_received = []
        stats_events = []
        done_event = None

        for line in raw_stream.split(b"\n"):
            line_str = line.decode("utf-8", errors="ignore").strip()
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            try:
                evt = json.loads(data_str)
                if evt.get("type") == "token":
                    tokens_received.append(evt.get("content"))
                elif evt.get("type") == "stats":
                    stats_events.append(evt)
                elif evt.get("type") == "done":
                    done_event = evt
            except Exception:
                continue

        assert len(tokens_received) > 0, "Expected streamed tokens in response"
        assert len(stats_events) > 0, "Expected in-flight stats telemetry events"
        assert done_event is not None, "Expected done event"
        assert done_event["ttft_ms"] > 0, "Expected positive TTFT"
        assert done_event["tokens"] == len(tokens_received)
        assert done_event["tps"] > 0, "Expected positive tokens/sec throughput"
        assert done_event["total_ms"] > 0, "Expected positive turnaround duration"

    finally:
        if os.path.exists(test_db):
            os.remove(test_db)
        if "FULCRUM_DB_PATH" in os.environ:
            del os.environ["FULCRUM_DB_PATH"]

    print("chat_web.py: PASSED")


def test_check_quota() -> None:
    print("Testing check_quota.py...")
    run(".venv/bin/python3 fulcrum/scripts/check_quota.py --help")

    from fulcrum.scripts.check_quota import check_accelerator_quota, format_quota_report

    # 1. Test GPU sufficient quota
    mock_gpu_good = [{"metric": "NVIDIA_L4_GPUS", "limit": 8.0, "usage": 2.0}]
    rep1 = check_accelerator_quota("test-proj", "us-central1", "vertex", "nvidia-l4", 2, mock_quotas=mock_gpu_good)
    assert rep1["status"] == "SUFFICIENT"
    assert rep1["available"] == 6.0
    assert rep1["limit"] == 8.0
    assert rep1["usage"] == 2.0
    fmt1 = format_quota_report(rep1)
    assert "SUFFICIENT" in fmt1
    assert "NVIDIA L4" in fmt1

    # 2. Test GPU insufficient quota
    mock_gpu_low = [{"metric": "NVIDIA_L4_GPUS", "limit": 2.0, "usage": 2.0}]
    rep2 = check_accelerator_quota("test-proj", "us-central1", "vertex", "nvidia-l4", 1, mock_quotas=mock_gpu_low)
    assert rep2["status"] == "INSUFFICIENT"
    assert rep2["available"] == 0.0
    fmt2 = format_quota_report(rep2)
    assert "INSUFFICIENT" in fmt2
    assert "Quota Console URL:" in fmt2

    # 3. Test Cloud TPU sufficient quota
    mock_tpu_good = [{"metric": "TPU_V6E_CHIPS", "limit": 16.0, "usage": 4.0}]
    rep3 = check_accelerator_quota("test-proj", "us-central1", "vertex", "v6e", 4, is_tpu=True, mock_quotas=mock_tpu_good)
    assert rep3["status"] == "SUFFICIENT"
    assert rep3["available"] == 12.0
    assert "Cloud TPU v6e" in rep3["accelerator"]

    # 4. Test Cloud TPU insufficient quota
    mock_tpu_low = [{"metric": "TPU_V5_LITE_PODSLICE_CHIPS", "limit": 4.0, "usage": 4.0}]
    rep4 = check_accelerator_quota("test-proj", "us-central1", "vertex", "v5e", 4, is_tpu=True, mock_quotas=mock_tpu_low)
    assert rep4["status"] == "INSUFFICIENT"
    assert rep4["available"] == 0.0

    # 5. Test unverified quota (empty response or metric not found)
    rep5 = check_accelerator_quota("test-proj", "us-central1", "vertex", "nvidia-h100-80gb", 8, mock_quotas=[])
    assert rep5["status"] == "UNVERIFIED"
    assert rep5["available"] is None

    # 6. Verify launch_gcp integrates quota check
    p_check = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model google/gemma-3-12b-it --format json")
    data_check = json.loads(p_check.stdout)
    assert "quota_check" in data_check

    p_no_check = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model google/gemma-3-12b-it --no-check-quota --format json")
    data_no_check = json.loads(p_no_check.stdout)
    assert data_no_check["quota_check"] is None

    print("check_quota.py: PASSED")


def main() -> None:
    print("Commencing complete /fulcrum test suite verification...\n")
    test_db_operations()
    test_find_model()
    test_launch_gcp()
    test_status_deployment()
    test_optimise_deployment()
    test_delete_deployment()
    test_benchmark_deployment()
    test_chat_web()
    test_check_quota()
    print("\nAll /fulcrum skill tests PASSED successfully!")


if __name__ == "__main__":
    main()
