"""Synthetic CUDA/resource gates and local hashes; no training or model loading."""

from copy import deepcopy
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import socket
import training.hardware as hw
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training.config import DEPENDENCY_PINS, load_config
from training.hardware import check_resources, inspect_hardware
from test_inference_planning import fixture_model


PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT.parents[1] / "work"


def ready_hardware():
    return {"packages": deepcopy(DEPENDENCY_PINS), "torch_probe_requested": True,
            "torch_probe_status": "completed", "cuda_available": True,
            "cuda_bfloat16_supported": True, "cuda_memory_total_bytes": 32 * 2**30,
            "cuda_memory_free_bytes": 24 * 2**30}


class HardwareInspectionTests(unittest.TestCase):
    def inventory(self, *, command=None, **kwargs):
        missing = importlib.metadata.PackageNotFoundError
        with patch.object(hw.platform, "system", return_value="Darwin"), \
             patch.object(hw.platform, "machine", return_value="arm64"), \
             patch.object(hw, "_command", side_effect=command or (lambda args: None)), \
             patch.object(hw, "_system_memory", return_value=16 * 2**30), \
             patch.object(hw.shutil, "which", return_value=None), \
             patch.object(hw.importlib.metadata, "version", side_effect=missing):
            return inspect_hardware(**kwargs)

    def test_plain_inspection_does_not_import_torch_and_separates_unified_memory(self):
        with patch.object(hw.importlib, "import_module", side_effect=AssertionError("no torch import")) as importer:
            result = self.inventory()
        self.assertEqual(importer.call_count, 0)
        self.assertEqual(result["unified_memory_total_bytes"], 16 * 2**30)
        self.assertIsNone(result["cuda_memory_total_bytes"])
        self.assertIsNone(result["cuda_memory_free_bytes"])
        self.assertFalse(result["cuda_available"])
        self.assertFalse(result["nvidia_smi_available"])
        self.assertEqual(result["torch_probe_status"], "not_requested")

    def test_mac_gpu_fields_are_filtered_and_serial_information_never_returned(self):
        payload = {"SPDisplaysDataType": [{"sppci_model": "Apple M1", "sppci_cores": "7",
                   "serial-number": "SYNTHETIC_SECRET_SERIAL", "spdisplays_ndrvs": [{"serial": "SYNTHETIC_DISPLAY_SERIAL"}]}]}
        def command(args):
            return json.dumps(payload) if args[0] == "system_profiler" else "Apple M1\n"
        result = self.inventory(command=command)
        self.assertEqual(result["metal_gpu_name"], "Apple M1")
        self.assertEqual(result["metal_gpu_cores"], 7)
        self.assertNotIn("SYNTHETIC_SECRET", json.dumps(result))
        self.assertNotIn("SYNTHETIC_DISPLAY", json.dumps(result))

    def test_optional_probe_reports_missing_torch_without_attempting_installation(self):
        with patch.object(hw.importlib, "import_module", side_effect=ImportError("synthetic absence")) as importer:
            result = self.inventory(probe_torch=True)
        importer.assert_called_once_with("torch")
        self.assertEqual(result["torch_probe_status"], "missing_torch")
        self.assertFalse(result["model_downloaded"])

    def fake_torch(self, *, available=True, capability=(8, 0), bf16=True):
        cuda = SimpleNamespace(is_available=lambda: available, get_device_properties=lambda index: SimpleNamespace(name="Synthetic CUDA GPU"),
            mem_get_info=lambda index: (24 * 2**30, 32 * 2**30), get_device_capability=lambda index: capability,
            device=lambda index: nullcontext(),
            is_bf16_supported=lambda *, including_emulation: bf16 if not including_emulation else True)
        return SimpleNamespace(__version__="2.9.0", version=SimpleNamespace(cuda="12.8"), cuda=cuda)

    def test_explicit_probe_records_cuda_zero_free_memory_and_native_bf16(self):
        with patch.object(hw.importlib, "import_module", return_value=self.fake_torch()):
            result = self.inventory(probe_torch=True)
        self.assertEqual(result["torch_probe_status"], "completed")
        self.assertTrue(result["cuda_available"])
        self.assertTrue(result["cuda_bfloat16_supported"])
        self.assertEqual(result["cuda_device_index"], 0)
        self.assertEqual(result["cuda_memory_free_bytes"], 24 * 2**30)
        self.assertEqual(result["cuda_memory_total_bytes"], 32 * 2**30)

    def test_emulated_bf16_or_cpu_backend_never_passes_native_cuda_gate(self):
        for backend in (self.fake_torch(capability=(7, 5)), self.fake_torch(bf16=False), self.fake_torch(available=False)):
            with patch.object(hw.importlib, "import_module", return_value=backend):
                result = self.inventory(probe_torch=True)
            self.assertFalse(result["cuda_bfloat16_supported"])

    def test_probe_uses_one_explicit_cuda_device_for_memory_and_bf16(self):
        backend = self.fake_torch()
        seen = []
        backend.cuda.mem_get_info = lambda index: (seen.append(("memory", index)) or (24 * 2**30, 32 * 2**30))
        backend.cuda.device = lambda index: (seen.append(("bf16", index)) or nullcontext())
        with patch.object(hw.importlib, "import_module", return_value=backend):
            result = self.inventory(probe_torch=True)
        self.assertEqual(result["torch_probe_status"], "completed")
        self.assertEqual(seen, [("memory", 0), ("bf16", 0)])

    def test_rocm_compatibility_api_does_not_count_as_cuda(self):
        backend = self.fake_torch(); backend.version.cuda = None
        with patch.object(hw.importlib, "import_module", return_value=backend):
            result = self.inventory(probe_torch=True)
        self.assertFalse(result["cuda_available"])
        self.assertFalse(result["cuda_bfloat16_supported"])
        self.assertIsNone(result["cuda_memory_free_bytes"])

    def test_probe_failure_has_sanitized_error_and_no_model_import(self):
        with patch.object(hw.importlib, "import_module", side_effect=RuntimeError("SYNTHETIC_SECRET_PATH")):
            result = self.inventory(probe_torch=True)
        self.assertEqual(result["torch_probe_status"], "failed")
        self.assertEqual(result["probe_error"], "torch_import_failed:RuntimeError")
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertFalse(result["model_imported"])

    def test_non_boolean_probe_option_is_rejected(self):
        with self.assertRaises(ValueError): inspect_hardware(probe_torch=1)


class ResourceGateTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(PROJECT / "configs/training/pilot.toml")
        self.model = fixture_model(identifier=self.config["model_id"], size=1e-9)
        self.model["enabled"] = True
        self.config["enable_execution"] = True
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-training-assets-", dir=WORK)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.weights = self.base / "weights"; self.weights.mkdir()
        (self.weights / "config.json").write_text('{"synthetic_fixture":true}')
        (self.weights / "model.safetensors").write_bytes(b"EXPLICIT SYNTHETIC TEST FIXTURE; NOT MODEL WEIGHTS")
        self.manifest_path = self.base / "weights-manifest.json"
        self.config["runtime"].update(weights_path=str(self.weights), weights_manifest=str(self.manifest_path))
        self.hardware = ready_hardware()
        self.publish_manifest()

    def publish_manifest(self):
        files = [{"path": str(path.relative_to(self.weights)), "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(self.weights.rglob("*")) if path.is_file()]
        manifest = {"schema_version": 1, "model_id": self.model["model_id"], "revision": self.model["revision"],
            "tokenizer_id": self.model["tokenizer"]["id"], "tokenizer_revision": self.model["tokenizer"]["revision"], "files": files}
        self.manifest_path.write_text(json.dumps(manifest))
        return manifest

    def result(self):
        return check_resources(self.config, self.model, self.hardware)

    def test_explicit_synthetic_ready_resources_pass_without_loading_a_model(self):
        with patch.object(hw.importlib, "import_module", side_effect=AssertionError("no backend import")), \
             patch.object(socket, "create_connection", side_effect=AssertionError("no network")):
            result = self.result()
        self.assertTrue(result["can_execute"])
        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["weights"]["status"], "hash_verified_local_bundle")
        self.assertIsNone(result["actual_gpu_hours"])
        self.assertIsNone(result["actual_cost"])
        self.assertFalse(result["model_downloaded"])

    def test_execution_and_registry_disable_flags_remain_separate_blockers(self):
        self.config["enable_execution"] = False
        self.model["enabled"] = False
        result = self.result()
        self.assertIn("execution_not_enabled", result["blockers"])
        self.assertIn("model_registry_entry_disabled", result["blockers"])
        self.assertFalse(result["can_execute"])

    def test_unified_ram_cannot_substitute_for_missing_cuda_memory(self):
        self.hardware.update(cuda_available=False, cuda_bfloat16_supported=None,
            cuda_memory_free_bytes=None, unified_memory_total_bytes=128 * 2**30)
        result = self.result()
        self.assertIn("cuda_free_memory_unverified", result["blockers"])
        self.assertIn("cuda_unavailable_or_unverified", result["blockers"])

    def test_hardware_must_have_an_explicit_completed_runtime_probe(self):
        self.hardware["torch_probe_requested"] = False
        self.assertIn("cuda_runtime_not_explicitly_verified", self.result()["blockers"])

    def test_declared_minimum_and_exact_integer_free_memory_are_checked(self):
        for free, blocker in ((8 * 2**30, "cuda_free_memory_below_declared_minimum"),
                              (None, "cuda_free_memory_unverified"), (True, "cuda_free_memory_unverified")):
            self.hardware["cuda_memory_free_bytes"] = free
            self.assertIn(blocker, self.result()["blockers"])

    def test_dependency_pins_are_exact_no_silent_torch_build_aliases(self):
        self.hardware["packages"]["torch"] = "2.9.0+cu128"
        self.hardware["packages"]["peft"] = None
        result = self.result()
        self.assertIn("dependency_version_mismatch:torch", result["blockers"])
        self.assertIn("missing_dependency:peft", result["blockers"])

    def test_malformed_package_inventory_is_rejected_explicitly(self):
        self.hardware["packages"] = None
        with self.assertRaisesRegex(ValueError, "version mapping"):
            self.result()

    def test_memory_estimate_is_only_two_bytes_per_declared_base_parameter(self):
        self.model["parameters_billion"] = 1.54
        result = self.result()
        estimates = result["memory_estimates"]
        self.assertEqual(estimates["base_weight_bytes_lower_bound"], 3_080_000_000)
        for field in ("activation_bytes", "optimizer_state_bytes", "rollout_cache_bytes", "trainable_adapter_bytes"):
            self.assertIsNone(estimates[field])
        self.assertIn("lower bound", result["warnings"][1])
        self.assertIn("invalid_local_weights", result["blockers"])  # Tiny fixtures are not full 1.54B weights.

    def test_empty_or_missing_local_paths_do_not_trigger_downloads(self):
        for field in ("weights_path", "weights_manifest"):
            cfg = deepcopy(self.config); cfg["runtime"][field] = ""
            result = check_resources(cfg, self.model, self.hardware)
            self.assertFalse(result["weights"]["verified"])
            self.assertIn("missing_" + field, result["blockers"])

    def test_raw_weight_corruption_is_detected(self):
        (self.weights / "model.safetensors").write_bytes(b"changed fixture")
        result = self.result()
        self.assertEqual(result["weights"]["status"], "invalid_local_weights")
        self.assertIn("hash/size", result["weights"]["reason"])

    def test_model_and_tokenizer_revision_mismatches_are_rejected(self):
        for field in ("model_id", "revision", "tokenizer_id", "tokenizer_revision"):
            manifest = self.publish_manifest(); manifest[field] += "-different"
            self.manifest_path.write_text(json.dumps(manifest))
            result = self.result()
            self.assertFalse(result["weights"]["verified"])
            self.assertIn("identity", result["weights"]["reason"])

    def test_duplicate_json_keys_boolean_schema_and_unsafe_file_paths_are_rejected(self):
        manifest = self.publish_manifest(); manifest["schema_version"] = True
        self.manifest_path.write_text(json.dumps(manifest))
        self.assertFalse(self.result()["weights"]["verified"])
        manifest = self.publish_manifest(); manifest["files"][0]["path"] = "../outside.json"
        self.manifest_path.write_text(json.dumps(manifest))
        self.assertFalse(self.result()["weights"]["verified"])
        self.manifest_path.write_text('{"schema_version":1,"schema_version":1}')
        self.assertFalse(self.result()["weights"]["verified"])

    def test_unlisted_local_assets_symlinks_and_missing_config_are_rejected(self):
        extra = self.weights / "unlisted.json"; extra.write_text("{}"); self.assertFalse(self.result()["weights"]["verified"])
        extra.unlink()
        linked = self.weights / "alias.safetensors"; linked.symlink_to(self.weights / "model.safetensors")
        self.assertFalse(self.result()["weights"]["verified"])
        linked.unlink(); (self.weights / "config.json").unlink(); self.publish_manifest()
        self.assertFalse(self.result()["weights"]["verified"])

    def test_minimum_memory_large_finite_value_stays_a_blocker_not_float_overflow(self):
        self.config["runtime"]["minimum_free_gpu_memory_gib"] = 1e300
        result = self.result()
        self.assertIn("cuda_free_memory_below_declared_minimum", result["blockers"])


if __name__ == "__main__":
    unittest.main()
