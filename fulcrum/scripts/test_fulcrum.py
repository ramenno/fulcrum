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
import unittest.mock as mock
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

    code_budget = """
from fulcrum.scripts.find_model import calculate_preflight_memory_budget, get_hardware_memory_gib

assert get_hardware_memory_gib('ct6e-standard-4t', 4) == 124.96
assert get_hardware_memory_gib('ct6e-standard-8t', 8) == 249.92
assert get_hardware_memory_gib('nvidia-l4', 1) == 24.0
assert get_hardware_memory_gib('nvidia-l4', 4) == 96.0

b_clamped = calculate_preflight_memory_budget('Qwen/Qwen2.5-32B-Instruct', 'ct6e-standard-4t', 4, 8192, is_tpu=True)
assert b_clamped['clamped'] is True
assert b_clamped['effective_max_model_len'] == 4096
assert b_clamped['gpu_memory_utilization'] == 0.90
assert b_clamped['max_safe_context'] < 8192

b_safe = calculate_preflight_memory_budget('Qwen/Qwen2.5-32B-Instruct', 'ct6e-standard-8t', 8, 8192, is_tpu=True)
assert b_safe['clamped'] is False
assert b_safe['effective_max_model_len'] == 8192
assert b_safe['max_safe_context'] > 8192
"""
    run(f'.venv/bin/python3 -c "{code_budget}"')

    # Test Strata MoE model search and 3-tier sizing
    p_strata = run(".venv/bin/python3 fulcrum/scripts/find_model.py --engine strata --moe-only --vram-limit 24 --format json")
    data_strata = json.loads(p_strata.stdout)
    assert len(data_strata) >= 4, "Expected at least 4 MoE models fitting 24GB VRAM via Strata"
    strata_ids = {item["metadata"]["id"] for item in data_strata}
    assert "Qwen/Qwen3.8-Flash-Next" in strata_ids
    assert "Qwen/Qwen3.6-35B-A3B" in strata_ids
    assert "mistralai/Mixtral-8x22B-Instruct-v0.1" in strata_ids
    assert "deepseek-ai/DeepSeek-V2-Lite-Chat" in strata_ids
    for item in data_strata:
        assert item["metadata"]["active_parameters_billion"] < item["metadata"]["parameters_billion"]
        hc = item["hardware_calc"]
        assert hc.get("is_moe") is True
        assert hc.get("recommended_strata") is not None
        assert hc["recommended_strata"]["vram"] == 24
        assert hc["recommended_strata"]["local_nvme_gb"] == 375
        assert "strata-cli serve" in hc.get("strata_command", "")

    p_strata_tbl = run(".venv/bin/python3 fulcrum/scripts/find_model.py --engine strata --moe-only --vram-limit 24")
    assert "STRATA GCE NODE (NVMe)" in p_strata_tbl.stdout
    assert "Strata MoE Node:" in p_strata_tbl.stdout
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
    p_tpu = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen2.5-14B-Instruct --tpu-type v6e --tpu-chips 4 --no-check-quota --format json")
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
    p_tpu_gcs = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen2.5-14B-Instruct --tpu-type v6e --tpu-chips 4 --no-check-quota --gcs-model-path gs://test-bucket/models/qwen --format json")
    data_tpu_gcs = json.loads(p_tpu_gcs.stdout)
    assert "GCS_MODEL_PATH=gs://test-bucket/models/qwen" in data_tpu_gcs["commands"]["upload_model"]
    assert "--model=/tmp/model" in data_tpu_gcs["commands"]["upload_model"]

    # 14. Test Cloud TPU Terraform generation
    p_tpu_tf = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen2.5-14B-Instruct --tpu-type v6e --tpu-chips 4 --no-check-quota --format terraform")
    assert 'machine_type      = "ct6e-standard-4t"' in p_tpu_tf.stdout
    assert "accelerator_type" not in p_tpu_tf.stdout
    assert "accelerator_count" not in p_tpu_tf.stdout
    assert 'image_uri     = "vllm/vllm-tpu:latest"' in p_tpu_tf.stdout

    # 15. Test pre-flight KV-cache memory budget calculation and automatic clamping
    p_clamped = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen2.5-32B-Instruct --tpu-type v6e --tpu-chips 4 --max-model-len 8192 --no-check-quota --format json")
    data_clamped = json.loads(p_clamped.stdout)
    assert "memory_budget" in data_clamped
    assert data_clamped["memory_budget"]["clamped"] is True
    assert data_clamped["memory_budget"]["effective_max_model_len"] == 4096
    assert data_clamped["memory_budget"]["gpu_memory_utilization"] == 0.90
    assert "--max-model-len=4096" in data_clamped["commands"]["upload_model"]
    assert "--gpu-memory-utilization=0.90" in data_clamped["commands"]["upload_model"]

    # 16. Test that 8-chip slice does not clamp 8192 context
    p_8t = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen2.5-32B-Instruct --tpu-type v6e --tpu-chips 8 --max-model-len 8192 --no-check-quota --format json")
    data_8t = json.loads(p_8t.stdout)
    assert data_8t["memory_budget"]["clamped"] is False
    assert data_8t["memory_budget"]["effective_max_model_len"] == 8192
    assert "--max-model-len=8192" in data_8t["commands"]["upload_model"]

    # 17. Test extract_operation_id_from_output helper
    from fulcrum.scripts.launch_gcp import (
        extract_operation_id_from_output,
        poll_vertex_operation,
        classify_vertex_error,
        VertexTransientCapacityError,
    )
    s1 = "Operation [projects/123/locations/europe-west4/operations/1810106235958067200] has not finished in 1800 seconds"
    assert extract_operation_id_from_output(s1) == "1810106235958067200"
    s2 = '{"name": "projects/my-proj/locations/us-central1/operations/998877665544332211"}'
    assert extract_operation_id_from_output(s2) == "998877665544332211"
    s3 = "1810106235958067200"
    assert extract_operation_id_from_output(s3) == "1810106235958067200"
    assert extract_operation_id_from_output("") is None
    assert extract_operation_id_from_output(None) is None

    # 18. Test poll_vertex_operation with mock responses
    # 18. Test classify_vertex_error differentiation
    c14 = classify_vertex_error(14, 'Machine type temporarily unavailable, please deploy with a different machine type or retry. Machine type: "ct6e-standard-4t"')
    assert c14["is_transient"] is True
    assert c14["quota_intact"] is True
    assert c14["category"] == "TRANSIENT_CAPACITY_STOCKOUT"
    assert "Code 14 UNAVAILABLE" in c14["title"]

    c14_msg = classify_vertex_error(None, "machine type temporarily unavailable")
    assert c14_msg["is_transient"] is True
    assert c14_msg["category"] == "TRANSIENT_CAPACITY_STOCKOUT"

    c8 = classify_vertex_error(8, "Resource exhausted: Quota exceeded for custom_model_serving_tpu_v6e")
    assert c8["is_transient"] is False
    assert c8["quota_intact"] is False
    assert c8["category"] == "QUOTA_EXHAUSTED"

    c3 = classify_vertex_error(3, "Invalid argument: accelerator_type is not supported on this machine type")
    assert c3["is_transient"] is False
    assert c3["quota_intact"] is True
    assert c3["category"] == "INVALID_CONFIGURATION"

    # 19. Test poll_vertex_operation with mock responses
    import unittest.mock as mock

    # Mock successful completion
    mock_success = mock.MagicMock()
    mock_success.returncode = 0
    mock_success.stdout = json.dumps({"done": True, "response": {"deployedModel": {"id": "12345"}}})

    with mock.patch("subprocess.run", return_value=mock_success):
        res = poll_vertex_operation("1810106235958067200", "test-proj", "europe-west4", timeout_seconds=10, interval_seconds=1, max_poll_iterations=1)
        assert res.get("done") is True

    # Mock failure error
    # Mock failure error (Quota Code 8)
    mock_fail = mock.MagicMock()
    mock_fail.returncode = 0
    mock_fail.stdout = json.dumps({"done": True, "error": {"code": 8, "message": "Resource exhausted"}})

    with mock.patch("subprocess.run", return_value=mock_fail):
        try:
            poll_vertex_operation("1810106235958067200", "test-proj", "europe-west4", timeout_seconds=10, interval_seconds=1, max_poll_iterations=1)
            assert False, "Expected RuntimeError on operation failure"
        except RuntimeError as e:
            assert "Resource exhausted" in str(e)

    # 19. Test async deploy command generated in launch_gcp dry-run
    # Mock transient capacity stock-out error (Code 14)
    mock_stockout = mock.MagicMock()
    mock_stockout.returncode = 0
    mock_stockout.stdout = json.dumps({
        "done": True,
        "error": {
            "code": 14,
            "message": 'Machine type temporarily unavailable, please deploy with a different machine type or retry. Machine type: "ct6e-standard-4t"',
        },
    })

    with mock.patch("subprocess.run", return_value=mock_stockout):
        try:
            poll_vertex_operation("1810106235958067200", "test-proj", "europe-west4", timeout_seconds=10, interval_seconds=1, max_poll_iterations=1)
            assert False, "Expected VertexTransientCapacityError on Code 14"
        except VertexTransientCapacityError as e:
            assert e.code == 14
            assert "temporarily unavailable" in str(e)
            assert "ct6e-standard-4t" in str(e)

    # 20. Test async deploy command generated in launch_gcp dry-run
    p_async_cmd = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model google/gemma-3-12b-it --format json")
    data_async = json.loads(p_async_cmd.stdout)
    deploy_cmd = data_async["commands"]["deploy_model"]
    assert "Option B: Asynchronous REST API dispatch" in deploy_cmd
    assert "gcloud ai operations describe" in deploy_cmd

    # 21. Test CLI flags --retry-transient, --max-retries, --retry-delay
    p_help = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --help")
    assert "--retry-transient" in p_help.stdout
    assert "--max-retries" in p_help.stdout
    assert "--retry-delay" in p_help.stdout
    assert "--engine" in p_help.stdout
    assert "--strata-ram" in p_help.stdout
    assert "--local-ssd" in p_help.stdout
    assert "--expert-evict" in p_help.stdout

    # 22. Test Strata MoE engine platform routing, machine sizing, and GCE NVMe provisioning
    from fulcrum.scripts.launch_gcp import evaluate_platform_viability, resolve_machine_type
    plat_s, rat_s = evaluate_platform_viability("Qwen/Qwen3.8-Flash-Next", "nvidia-l4", 1, "auto", engine="strata")
    assert plat_s == "gce"
    assert "Strata" in rat_s
    assert resolve_machine_type("nvidia-l4", 1, platform="gce", engine="strata", model_id="Qwen/Qwen3.8-Flash-Next") == "g2-standard-32"
    assert resolve_machine_type("nvidia-l4", 1, platform="gce", engine="strata", model_id="Qwen/Qwen3.6-35B-A3B") == "g2-standard-16"

    p_strata_gce = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen3.8-Flash-Next --engine strata --no-check-quota --format json")
    data_strata_gce = json.loads(p_strata_gce.stdout)
    assert data_strata_gce["engine"] == "strata"
    assert data_strata_gce["platform"] == "gce"
    assert data_strata_gce["machine_type"] == "g2-standard-32"
    assert data_strata_gce["local_ssd_count"] == 1
    assert "--local-ssd=interface=NVME" in data_strata_gce["create_command"]
    assert "strata-cli serve" in data_strata_gce["startup_script"]
    assert "--vram 22" in data_strata_gce["startup_script"]
    assert "--ram 96" in data_strata_gce["startup_script"]
    assert "--evict lfu" in data_strata_gce["startup_script"]

    p_strata_tf = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen3.8-Flash-Next --engine strata --no-check-quota --format terraform")
    assert 'scratch_disk {' in p_strata_tf.stdout
    assert 'interface = "NVME"' in p_strata_tf.stdout

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
    # Test query_vertex_operation with Code 14 mock
    from fulcrum.scripts.status_deployment import query_vertex_operation
    mock_stockout_op = mock.MagicMock()
    mock_stockout_op.returncode = 0
    mock_stockout_op.stdout = json.dumps({
        "done": True,
        "error": {
            "code": 14,
            "message": 'Machine type temporarily unavailable, please deploy with a different machine type or retry. Machine type: "ct6e-standard-4t"',
        },
    })
    with mock.patch("subprocess.run", return_value=mock_stockout_op):
        st, done, err_txt = query_vertex_operation("1810106235958067200", "europe-west4", "test-proj")
        assert st == "STOCKOUT"
        assert done is True
        assert "temporarily unavailable" in err_txt

    # Populate test database with Vertex, GCE, and STOCKOUT deployments
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
            "operation_id": "1810106235958067200",
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
        {
            "id": "fulcrum-stockout-tpu",
            "endpoint_name": "fulcrum-stockout-tpu",
            "platform": "vertex",
            "model": "Qwen/Qwen2.5-32B-Instruct",
            "project": "test-proj",
            "region": "europe-west4",
            "machine_type": "ct6e-standard-4t",
            "accelerator": "4x Cloud TPU V6E chips",
            "gpu_count": 4,
            "operation_id": "1810106235958067200",
            "operation_error": 'code: 14: Machine type temporarily unavailable, please deploy with a different machine type or retry. Machine type: "ct6e-standard-4t"',
            "status": "STOCKOUT",
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
    assert "STOCKOUT" in p_table.stdout

    # Test JSON output
    p_json = run(".venv/bin/python3 fulcrum/scripts/status_deployment.py --format json", env_vars={"FULCRUM_DB_PATH": test_db})
    data = json.loads(p_json.stdout)
    assert len(data) == 3
    assert data[0]["platform"] == "vertex"
    assert data[0]["operation_id"] == "1810106235958067200"
    assert data[1]["platform"] == "gce"
    assert data[2]["status"] == "STOCKOUT"

    # Test verbose diagnostics for Vertex (including Operation ID display)
    p_verb = run(".venv/bin/python3 fulcrum/scripts/status_deployment.py --id fulcrum-gemma-3-12b-it --verbose", env_vars={"FULCRUM_DB_PATH": test_db})
    assert "Vertex Predict URI" in p_verb.stdout
    assert "1810106235958067200" in p_verb.stdout

    # Test verbose diagnostics for STOCKOUT
    p_verb_so = run(".venv/bin/python3 fulcrum/scripts/status_deployment.py --id fulcrum-stockout-tpu --verbose", env_vars={"FULCRUM_DB_PATH": test_db})
    assert "Transient datacentre capacity stock-out" in p_verb_so.stdout
    assert "Quota is INTACT" in p_verb_so.stdout
    assert "europe-west4" in p_verb_so.stdout
    assert "Suggested Action" in p_verb_so.stdout

    # Test Cloud Logging log formatting and filter generation
    from fulcrum.scripts.status_deployment import (
        format_vllm_log_entry,
        build_vertex_log_filter,
        build_gce_log_filter,
        resolve_endpoint_numeric_id,
    )

    # 1. Format classification tests
    entry_core = {"timestamp": "2026-09-24T10:00:00Z", "severity": "INFO", "textPayload": "INFO 09-24 10:00:00 [core.py:123] Initializing an LLM engine (v0.6.2)..."}
    res_core = format_vllm_log_entry(entry_core)
    assert "[vLLM EngineCore]" in res_core
    assert "Initializing an LLM engine" in res_core

    entry_weights = {"timestamp": "2026-09-24T10:05:00Z", "severity": "INFO", "textPayload": "Fetching 15 safetensors checkpoint shards: 100% complete"}
    res_weights = format_vllm_log_entry(entry_weights)
    assert "[Weight Streaming]" in res_weights

    entry_jax = {"timestamp": "2026-09-24T10:12:00Z", "severity": "INFO", "textPayload": "Compiling graph for bucket 512 using Pallas kernel on Cloud TPU v6e..."}
    res_jax = format_vllm_log_entry(entry_jax)
    assert "[JAX/XLA Compilation]" in res_jax

    entry_api = {"timestamp": "2026-09-24T10:20:00Z", "severity": "INFO", "jsonPayload": {"message": "Route /v1/chat/completions registered and healthy on port 8080"}}
    res_api = format_vllm_log_entry(entry_api)
    assert "[vLLM APIServer]" in res_api

    entry_err = {"timestamp": "2026-09-24T10:25:00Z", "severity": "ERROR", "textPayload": "CUDA out of memory error"}
    res_err = format_vllm_log_entry(entry_err)
    assert "[Error]" in res_err

    # 2. Filter builder tests
    vf = build_vertex_log_filter("1810106235958067200", min_timestamp="2026-09-24T10:00:00Z")
    assert 'resource.type="aiplatform.googleapis.com/Endpoint"' in vf
    assert 'resource.labels.endpoint_id="1810106235958067200"' in vf
    assert 'timestamp > "2026-09-24T10:00:00Z"' in vf

    gf = build_gce_log_filter("my-instance")
    assert 'resource.type="gce_instance"' in gf

    # 3. Numeric ID resolver tests
    assert resolve_endpoint_numeric_id("1810106235958067200", "us-central1", "proj") == "1810106235958067200"

    # 4. CLI help test for --logs and --follow
    p_help = run(".venv/bin/python3 fulcrum/scripts/status_deployment.py --help")
    assert "--logs" in p_help.stdout
    assert "--follow" in p_help.stdout

    # 5. In-process fetch_cloud_logs and tail_deployment_logs test
    import io
    from contextlib import redirect_stdout
    from fulcrum.scripts.status_deployment import tail_deployment_logs, fetch_cloud_logs

    mock_log_proc = mock.MagicMock()
    mock_log_proc.returncode = 0
    mock_log_proc.stdout = json.dumps([
        {"timestamp": "2026-09-24T10:00:00Z", "severity": "INFO", "textPayload": "Initializing an LLM engine"},
        {"timestamp": "2026-09-24T10:05:00Z", "severity": "INFO", "textPayload": "Fetching safetensors checkpoint shards"},
    ])
    with mock.patch("subprocess.run", return_value=mock_log_proc):
        logs = fetch_cloud_logs('resource.type="aiplatform.googleapis.com/Endpoint"', "test-proj", 10)
        assert len(logs) == 2

        buf = io.StringIO()
        with redirect_stdout(buf):
            tail_deployment_logs(dummy_deployments[0], follow=False, lines=10, raw=False)
        out = buf.getvalue()
        assert "Tailing Cloud Logging for Vertex AI Endpoint" in out
        assert "[vLLM EngineCore]" in out
        assert "[Weight Streaming]" in out

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
    assert data_tpu_opt["flags"]["--gpu-memory-utilization"] == 0.90
    assert "--swap-space" not in data_tpu_opt["flags"]
    assert data_tpu_opt["flags"]["--block-size"] == 16
    assert any("Omit --swap-space" in r for r in data_tpu_opt["rationales"])
    assert any("Pre-Flight KV-Cache Sizing" in r for r in data_tpu_opt["rationales"])
    assert any("Pallas" in r for r in data_tpu_opt["rationales"])
    assert any("Static Sequence Length Bucketing" in r for r in data_tpu_opt["rationales"])
    assert any("Do NOT pass --accelerator" in r for r in data_tpu_opt["rationales"])

    # Test Strata MoE optimisation advice
    p_strata_opt = run(".venv/bin/python3 fulcrum/scripts/optimise_deployment.py advise --model Qwen/Qwen3.8-Flash-Next --engine strata --workload latency --format json")
    data_strata_opt = json.loads(p_strata_opt.stdout)
    assert data_strata_opt["engine"] == "strata"
    assert data_strata_opt["flags"]["--vram"] == 22
    assert data_strata_opt["flags"]["--ram"] == 96
    assert data_strata_opt["flags"]["--evict"] == "lfu"
    assert data_strata_opt["flags"]["--kv"] == "int8"
    assert data_strata_opt["flags"]["--spec"] == 3
    assert "strata-cli serve" in data_strata_opt["command"]
    assert any("3-Tier MoE Memory Hierarchy" in r for r in data_strata_opt["rationales"])
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

    from fulcrum.scripts.check_quota import (
        check_accelerator_quota,
        format_quota_report,
        format_tpu_sweep_report,
        sweep_tpu_quota,
        QUOTA_METRIC_MAP,
        normalize_accelerator_key,
    )

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

    # 6. Test explicit TPU generations in QUOTA_METRIC_MAP
    assert "custom_model_serving_tpu_v6e" in QUOTA_METRIC_MAP
    assert "custom_model_serving_tpu_v5e" in QUOTA_METRIC_MAP
    assert "custom_model_serving_tpu_v5p" in QUOTA_METRIC_MAP
    assert QUOTA_METRIC_MAP["custom_model_serving_tpu_v6e"]["vertex_metric"] == "custom_model_serving_tpu_v6e"
    assert QUOTA_METRIC_MAP["custom_model_serving_tpu_v5e"]["vertex_metric"] == "custom_model_serving_tpu_v5e"
    assert QUOTA_METRIC_MAP["custom_model_serving_tpu_v5p"]["vertex_metric"] == "custom_model_serving_tpu_v5p"
    assert normalize_accelerator_key("custom_model_serving_tpu_v6e") == "v6e"
    assert normalize_accelerator_key("custom_model_serving_tpu_v5e") == "v5e"
    assert normalize_accelerator_key("custom_model_serving_tpu_v5p") == "v5p"

    # 7. Test automated regional quota sweeping (reproducing user incident: us-central1=0, europe-west4=8)
    incident_mock = {
        "us-central1": 0.0,
        "europe-west4": 8.0,
        "asia-southeast1": 0.0,
        "us-east4": 0.0,
    }
    sweep_rep = sweep_tpu_quota("test-proj", tpu_gen="v6e", required_chips=8, mock_regional_quotas=incident_mock)
    assert sweep_rep["status"] == "SUFFICIENT"
    assert sweep_rep["selected_region"] == "europe-west4"
    assert sweep_rep["regional_headroom"]["europe-west4"] == 8.0
    assert sweep_rep["regional_headroom"]["us-central1"] == 0.0
    fmt_sweep = format_tpu_sweep_report(sweep_rep)
    assert "Proactive Multi-Region Cloud TPU Quota Discovery: SUFFICIENT" in fmt_sweep
    assert "europe-west4" in fmt_sweep
    assert "<-- Selected for Deployment" in fmt_sweep

    # 8. Test sweeping when all regions have 0 quota
    zero_mock = {
        "us-central1": 0.0,
        "europe-west4": 0.0,
        "asia-southeast1": 0.0,
        "us-east4": 0.0,
    }
    sweep_zero = sweep_tpu_quota("test-proj", tpu_gen="v6e", required_chips=4, mock_regional_quotas=zero_mock)
    assert sweep_zero["status"] == "INSUFFICIENT"
    assert "All audited TPU regions" in sweep_zero["message"]

    # 9. Verify launch_gcp integrates quota check
    p_check = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model google/gemma-3-12b-it --format json")
    data_check = json.loads(p_check.stdout)
    assert "quota_check" in data_check

    p_no_check = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model google/gemma-3-12b-it --no-check-quota --format json")
    data_no_check = json.loads(p_no_check.stdout)
    assert data_no_check["quota_check"] is None

    # 10. Verify launch_gcp honors explicit --region on TPU
    p_tpu_reg = run(".venv/bin/python3 fulcrum/scripts/launch_gcp.py --model Qwen/Qwen2.5-14B-Instruct --tpu-type v6e --tpu-chips 4 --region europe-west4 --format json")
    data_tpu_reg = json.loads(p_tpu_reg.stdout)
    assert data_tpu_reg["region"] == "europe-west4"

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
