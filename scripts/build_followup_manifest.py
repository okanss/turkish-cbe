"""Build the targeted follow-up subset (revision R1, Reviewer 2 comment 5a).

Subset of the frozen P1 expanded manifest: all nine relations, native_shared
names, EN canonical + TR canonical + TR P3, n in {4, 16, 64}. Rows are copied
verbatim (prompt text and SHA-256 unchanged), so the direct arm can be read
from the existing sweep records and only the structured-reasoning arm is new.
"""
import json, sys
from pathlib import Path
SRC = Path("data/manifests/p1_expanded_manifest.jsonl")
DST = Path("data/manifests/followup_reasoning_manifest.jsonl")
KEEP = {("en", "canonical"), ("tr", "canonical"), ("tr", "P3")}
SIZES = {4, 16, 64}
rows = [json.loads(l) for l in SRC.open(encoding="utf-8")]
sub = [r for r in rows if r["name_control"] == "native_shared"
       and (r["language"], r["condition"]) in KEEP and r["n"] in SIZES]
# keep only query cells that have all three conditions (fully matched)
from collections import defaultdict
cells = defaultdict(set)
for r in sub:
    cells[(r["query_instance_id"], r["relation_id"])].add((r["language"], r["condition"]))
sub = [r for r in sub if cells[(r["query_instance_id"], r["relation_id"])] == KEEP]
sub.sort(key=lambda r: r["prompt_id"])
if DST.exists():
    sys.exit(f"{DST} exists (write-once); delete it deliberately to rebuild")
with DST.open("w", encoding="utf-8") as f:
    for r in sub:
        f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
print(f"{len(sub)} prompts, {len({r['latent_instance_id'] for r in sub})} latent orders -> {DST}")
