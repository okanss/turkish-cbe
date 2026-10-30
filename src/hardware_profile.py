"""Apple Silicon / Ollama hardware profile for the P1 local evaluation.

Measures the machine and the local Ollama installation; never assumes CUDA,
cloud hardware, or that a model fits because its weights fit. Every number in
the emitted profile is read from the system or the Ollama API at run time,
with its collection method recorded alongside it.

Run from the project root:
    python3 src/hardware_profile.py --output reports/hardware_profile.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
GIB = 2**30

# Candidate models for the P1 evaluation (Ollama library tags; the qwen2.5
# chat tags are the instruct variants). Weight sizes are read from the local
# store after pull, never hardcoded.
CANDIDATES = ["qwen2.5:0.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"]

# Conservative feasibility rule, documented rather than assumed:
# weights + KV/runtime headroom must fit inside the memory the OS can
# actually free, and weights alone must not exceed 60% of total unified
# memory (GPU wired allocation limits on macOS sit below total RAM).
RUNTIME_HEADROOM_BYTES = 3 * GIB
WEIGHT_FRACTION_CAP = 0.60


def _run(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()


def _sysctl(name: str) -> str:
    return _run(["sysctl", "-n", name])


def _vm_stat() -> dict:
    raw = _run(["vm_stat"])
    page = int(re.search(r"page size of (\d+)", raw).group(1))
    values = {}
    for line in raw.splitlines()[1:]:
        if ":" in line:
            key, _, count = line.rpartition(":")
            values[key.strip()] = int(count.strip().rstrip(".")) * page
    return {"page_size": page, **values}


def _swap() -> dict:
    raw = _sysctl("vm.swapusage")
    fields = dict(re.findall(r"(\w+) = ([\d.]+)M", raw))
    return {k: float(v) * 2**20 for k, v in fields.items()}


def _memory_pressure() -> str | None:
    if shutil.which("memory_pressure") is None:
        return None
    raw = _run(["memory_pressure", "-Q"]) or _run(["memory_pressure"])
    match = re.search(r"System-wide memory free percentage: (\d+)%", raw)
    return match.group(1) + "% free" if match else None


def _ollama(path: str):
    with urllib.request.urlopen(OLLAMA_URL + path, timeout=10) as response:
        return json.loads(response.read())


def collect() -> dict:
    vm = _vm_stat()
    total = int(_sysctl("hw.memsize"))
    # Reclaimable now: free + inactive + purgeable + file-backed. This is an
    # estimate of what the OS can hand to Ollama, not a guarantee.
    available = sum(vm.get(k, 0) for k in (
        "Pages free", "Pages inactive", "Pages purgeable", "File-backed pages"))
    disk = shutil.disk_usage(Path.home())
    store = Path.home() / ".ollama/models"
    store_target = store.resolve()
    store_disk = shutil.disk_usage(store_target) if store_target.exists() else None
    try:
        version = _ollama("/api/version")["version"]
        models = [{
            "name": m["name"], "digest": m["digest"],
            "size_bytes": m["size"],
            "quantization": m.get("details", {}).get("quantization_level"),
            "parameter_size": m.get("details", {}).get("parameter_size"),
            "format": m.get("details", {}).get("format"),
        } for m in _ollama("/api/tags")["models"]]
        server = True
    except OSError:
        version, models, server = None, [], False
    profile = {
        "collected_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
        "is_apple_silicon": platform.machine() == "arm64",
        "cpu_cores": int(_sysctl("hw.ncpu")),
        "performance_cores": int(_sysctl("hw.perflevel0.physicalcpu") or 0),
        "efficiency_cores": int(_sysctl("hw.perflevel1.physicalcpu") or 0),
        "unified_memory_total_bytes": total,
        "unified_memory_reclaimable_now_bytes": available,
        "reclaimable_method": "vm_stat free+inactive+purgeable+file-backed; estimate, not a guarantee",
        "memory_pressure": _memory_pressure(),
        "swap": _swap(),
        "macos_version": _run(["sw_vers", "-productVersion"]),
        "home_disk_free_bytes": disk.free,
        "ollama_model_store": str(store_target),
        "ollama_store_disk_free_bytes": store_disk.free if store_disk else None,
        "ollama_server_running": server,
        "ollama_version": version,
        "installed_models": models,
    }
    by_name = {m["name"]: m for m in models}
    feasibility = []
    for tag in CANDIDATES:
        entry = by_name.get(tag)
        if entry is None:
            feasibility.append({"model": tag, "installed": False, "verdict": "not_installed"})
            continue
        weights = entry["size_bytes"]
        fits_cap = weights <= WEIGHT_FRACTION_CAP * total
        fits_now = weights + RUNTIME_HEADROOM_BYTES <= available
        verdict = ("feasible" if fits_cap and fits_now
                   else "marginal_requires_measurement" if fits_cap
                   else "exceeds_weight_fraction_cap")
        feasibility.append({
            "model": tag, "installed": True, "digest": entry["digest"],
            "quantization": entry["quantization"], "weights_bytes": weights,
            "weight_fraction_of_total": round(weights / total, 3),
            "fits_weight_cap": fits_cap, "fits_reclaimable_now": fits_now,
            "verdict": verdict,
            "note": "Static rule only; the Phase 6 speed benchmark measures the real fit.",
        })
    profile["candidate_feasibility"] = feasibility
    return profile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/hardware_profile.json"))
    args = parser.parse_args(argv)
    profile = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: profile[k] for k in (
        "cpu_brand", "unified_memory_total_bytes", "unified_memory_reclaimable_now_bytes",
        "ollama_version", "ollama_server_running")}, indent=1))
    for row in profile["candidate_feasibility"]:
        print(row["model"], "→", row["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
