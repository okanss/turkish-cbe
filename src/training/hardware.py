"""Read-only hardware inventory and conservative local training preconditions."""

from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess

from inference.config import validate_model
from .config import DEPENDENCY_PINS, validate_config


def _command(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _system_memory():
    if platform.system() == "Darwin":
        value = _command(["sysctl", "-n", "hw.memsize"])
        if value and value.strip().isdigit():
            return int(value.strip())
    try:
        size = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return int(size) if size > 0 else None
    except (ValueError, OSError, AttributeError):
        return None


def inspect_hardware(*, probe_torch=False):
    """Inventory metadata without importing torch unless explicitly requested.

    Only selected CPU/GPU names, memory amounts, package versions and backend
    capabilities are retained; system-profiler serial/display data and secrets
    are never returned. Unified Apple memory is distinct from CUDA memory.
    The optional probe initializes the CUDA runtime, never a model or download.
    """
    if type(probe_torch) is not bool:
        raise ValueError("probe_torch must be an exact Boolean")
    system, machine = platform.system(), platform.machine()
    processor = None
    metal_name = metal_cores = None
    if system == "Darwin":
        text = _command(["sysctl", "-n", "machdep.cpu.brand_string"])
        processor = text.strip() if text else None
        display = _command(["system_profiler", "SPDisplaysDataType", "-json"])
        if display:
            try:
                devices = json.loads(display).get("SPDisplaysDataType", [])
                for device in devices:
                    name = device.get("sppci_model") or device.get("spdisplays_chipset-model")
                    if isinstance(name, str) and "Apple" in name:
                        metal_name = name
                        cores = str(device.get("sppci_cores", ""))
                        metal_cores = int(cores) if cores.isdigit() else None
                        break
            except (ValueError, TypeError, AttributeError):
                pass
    total = _system_memory()
    unified = total if system == "Darwin" and machine in {"arm64", "aarch64"} else None
    versions = {}
    for name in DEPENDENCY_PINS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    smi = shutil.which("nvidia-smi")
    nvidia = []
    if smi:
        output = _command([smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
        if output:
            for row in csv.reader(io.StringIO(output)):
                try:
                    name, memory, free = [value.strip() for value in row]
                    nvidia.append({"name": name, "total_memory_bytes": int(memory) * 2**20,
                                   "free_memory_bytes": int(free) * 2**20})
                except (ValueError, TypeError):
                    continue
    result = {"schema_version": 1, "system": system, "machine": machine,
              "python": platform.python_version(), "processor": processor,
              "system_memory_total_bytes": total, "unified_memory_total_bytes": unified,
              "metal_gpu_name": metal_name, "metal_gpu_cores": metal_cores,
              "nvidia_smi_available": smi is not None, "nvidia_gpus": nvidia, "packages": versions,
              "torch_probe_requested": probe_torch, "torch_probe_status": "not_requested",
              "cuda_available": False if system == "Darwin" else None,
              "cuda_bfloat16_supported": None, "cuda_device_index": None, "cuda_device_name": None,
              "cuda_memory_total_bytes": None, "cuda_memory_free_bytes": None,
              "cuda_runtime_version": None, "torch_runtime_version": None,
              "probe_error": None, "model_imported": False, "model_downloaded": False}
    if not probe_torch:
        return result
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        result.update(torch_probe_status="missing_torch", probe_error="torch_not_importable")
        return result
    except Exception as error:
        result.update(torch_probe_status="failed", probe_error="torch_import_failed:" + type(error).__name__)
        return result
    try:
        result["torch_runtime_version"] = str(torch.__version__)
        result["cuda_runtime_version"] = torch.version.cuda
        # ROCm also exposes torch.cuda; this protocol requires an NVIDIA CUDA
        # build and must not mistake that compatibility API for CUDA itself.
        result["cuda_available"] = bool(torch.cuda.is_available()) and bool(torch.version.cuda)
        if result["cuda_available"]:
            properties = torch.cuda.get_device_properties(0)
            free, memory = torch.cuda.mem_get_info(0)
            capability = torch.cuda.get_device_capability(0)
            # Native CUDA BF16 requires Ampere-or-newer capability; emulated
            # support must not silently satisfy the requested training backend.
            native = tuple(capability) >= (8, 0)
            with torch.cuda.device(0):
                supported = bool(torch.cuda.is_bf16_supported(including_emulation=False))
            result.update(cuda_device_index=0, cuda_device_name=str(properties.name),
                cuda_memory_free_bytes=int(free), cuda_memory_total_bytes=int(memory),
                cuda_bfloat16_supported=native and supported)
        else:
            result["cuda_bfloat16_supported"] = False
        result["torch_probe_status"] = "completed"
    except Exception as error:
        result.update(torch_probe_status="failed", probe_error="cuda_probe_failed:" + type(error).__name__)
    return result


def _decode(data):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate weights manifest key")
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError("nonfinite weights manifest value")
    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def _file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _weights(config, model, minimum_bytes):
    runtime = config["runtime"]
    if not runtime["weights_path"]:
        return {"status": "missing_weights_path", "verified": False, "payload_bytes": None}
    if not runtime["weights_manifest"]:
        return {"status": "missing_weights_manifest", "verified": False, "payload_bytes": None}
    root, manifest_path = Path(runtime["weights_path"]), Path(runtime["weights_manifest"])
    if not root.is_dir() or not manifest_path.is_file():
        return {"status": "missing_local_weights_or_manifest", "verified": False, "payload_bytes": None}
    try:
        if any(path.is_symlink() for path in (root, *root.parents, manifest_path, *manifest_path.parents)):
            raise ValueError("symlinked weights are not a frozen local bundle")
        data = manifest_path.read_bytes()
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("weights manifest exceeds 4 MiB")
        manifest = _decode(data)
        expected = {"schema_version", "model_id", "revision", "tokenizer_id", "tokenizer_revision", "files"}
        if not isinstance(manifest, dict) or set(manifest) != expected:
            raise ValueError("weights manifest schema mismatch")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise ValueError("weights manifest version mismatch")
        for key, value in (("model_id", model["model_id"]), ("revision", model["revision"]),
                           ("tokenizer_id", model["tokenizer"]["id"]), ("tokenizer_revision", model["tokenizer"]["revision"])):
            if manifest[key] != value or not isinstance(manifest[key], str):
                raise ValueError("weights manifest model/tokenizer identity mismatch")
        files = manifest["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("weights manifest has no files")
        seen = set(); payload = 0; weight_names = []
        for item in files:
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
                raise ValueError("weights file schema mismatch")
            relative = item["path"]
            if (type(relative) is not str or "\\" in relative or PurePosixPath(relative).is_absolute()
                    or ".." in PurePosixPath(relative).parts or relative in seen or relative in {"", "."}
                    or PurePosixPath(relative).as_posix() != relative):
                raise ValueError("unsafe or duplicate weights path")
            if type(item["bytes"]) is not int or item["bytes"] < 0 or not isinstance(item["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
                raise ValueError("invalid weights size/hash")
            path = root / relative
            if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError("missing or symlinked weights file")
            if path.stat().st_size != item["bytes"] or _file_hash(path) != item["sha256"]:
                raise ValueError("weights payload hash/size mismatch")
            seen.add(relative)
            name = path.name
            if (name.startswith("model") and name.endswith(".safetensors")) or (name.startswith("pytorch_model") and name.endswith(".bin")):
                weight_names.append(name); payload += item["bytes"]
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError("unlisted symlink in local weights bundle")
        actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path != manifest_path}
        if actual != seen or "config.json" not in seen or not weight_names:
            raise ValueError("every local asset and config must be bound, including full model weights")
        if minimum_bytes is not None and payload < minimum_bytes:
            raise ValueError("weight payload is smaller than the declared BF16 parameter lower bound")
        return {"status": "hash_verified_local_bundle", "verified": True, "payload_bytes": payload,
                "manifest_sha256": hashlib.sha256(data).hexdigest(), "n_files": len(files),
                "identity_scope": "declared immutable identity plus local hashes; tensor compatibility still requires model loading"}
    except (ValueError, OSError, TypeError) as error:
        return {"status": "invalid_local_weights", "verified": False, "payload_bytes": None, "reason": str(error)}


def check_resources(config, model, hardware):
    """Return blockers and labelled memory lower bounds, never a fit guarantee."""
    validate_config(config); validate_model(model)
    if not isinstance(hardware, dict):
        raise ValueError("hardware must be an inventory mapping")
    blockers = []
    if not config["enable_execution"]: blockers.append("execution_not_enabled")
    if model["id"] != config["model_id"]: blockers.append("selected_model_registry_identity_mismatch")
    if model["provider"] != "transformers": blockers.append("training_requires_local_transformers_backend")
    if not model["enabled"]: blockers.append("model_registry_entry_disabled")
    if not isinstance(model["revision"], str) or not re.fullmatch(r"[a-f0-9]{40}", model["revision"]): blockers.append("model_revision_not_immutable")
    if hardware.get("torch_probe_status") != "completed" or hardware.get("torch_probe_requested") is not True:
        blockers.append("cuda_runtime_not_explicitly_verified")
    if hardware.get("cuda_available") is not True: blockers.append("cuda_unavailable_or_unverified")
    if hardware.get("cuda_bfloat16_supported") is not True: blockers.append("native_cuda_bfloat16_unavailable_or_unverified")
    minimum = math.ceil(Fraction(str(config["runtime"]["minimum_free_gpu_memory_gib"])) * 2**30)
    free = hardware.get("cuda_memory_free_bytes")
    if type(free) is not int or free < 0:
        blockers.append("cuda_free_memory_unverified")
    elif free < minimum:
        blockers.append("cuda_free_memory_below_declared_minimum")
    packages = hardware.get("packages", {})
    if not isinstance(packages, dict):
        raise ValueError("hardware packages must be a version mapping")
    for name, required in config["dependencies"].items():
        actual = packages.get(name)
        if actual is None: blockers.append("missing_dependency:" + name)
        elif actual != required: blockers.append("dependency_version_mismatch:" + name)
    parameters = model["parameters_billion"]
    weight_lower = math.ceil(Fraction(str(parameters)) * 2_000_000_000) if parameters is not None else None
    if weight_lower is None: blockers.append("model_parameter_count_unavailable")
    elif type(free) is int and free >= 0 and free < weight_lower: blockers.append("cuda_free_memory_below_base_weight_lower_bound")
    weights = _weights(config, model, weight_lower)
    if not weights["verified"]: blockers.append(weights["status"])
    return {"can_execute": not blockers, "status": "eligible_for_explicit_execution" if not blockers else "blocked",
            "blockers": sorted(set(blockers)), "weights": weights,
            "memory_estimates": {"base_weight_bytes_lower_bound": weight_lower,
                "bytes_per_base_parameter_assumed": 2, "minimum_free_cuda_bytes_declared": minimum,
                "activation_bytes": None, "optimizer_state_bytes": None, "rollout_cache_bytes": None,
                "trainable_adapter_bytes": None, "estimate_scope": "BF16 base weights only; not a total-memory fit prediction"},
            "warnings": ["Unified/system RAM is not counted as CUDA free memory.",
                "Declared free-memory minimum and base-weight lower bound do not guarantee training will fit.",
                "This pinned software stack still needs tensor/model compatibility and runtime training validation."],
            "actual_gpu_hours": None, "actual_cost": None, "model_downloaded": False}
