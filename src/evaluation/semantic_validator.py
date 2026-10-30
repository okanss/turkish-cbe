"""P1 automatic semantic re-verification of the frozen evaluation manifest.

For every sampled prompt this validator re-derives, from the surface text
alone, the structural facts the benchmark asserts — using the project's
closed-grammar normalizers (``validation.semantic_equivalence``) and the
exact graph oracle (``data_generation.oracle``), never an LLM judge.

Checks per prompt:
  1. premise edges parsed from the text equal the latent order's adjacent
     chain exactly (no omitted entities, no extra relations);
  2. the parsed query matches the recorded subject/object and the normalized
     relation direction;
  3. the graph oracle's answer over the parsed edges equals the stored gold
     label (logical order and expected answer preserved);
  4. entity references are consistent (surface inventory == latent inventory).
Checks per matched group (query x relation x name control):
  5. identical gold label and identical normalized semantic digest across all
     presentation cells (transformation equivalence);
  6. the condition label is one of the registered transformations.

A prompt whose surface form the closed grammar cannot parse is NOT treated as
invalid: it is recorded as ``not_normalizable`` — exactly the residue that the
human linguistic review must adjudicate. Automatic passes certify agreement
with the implemented grammar only, never naturalness (manuscript Section 3.3).

Run:  PYTHONPATH=src python3 -m evaluation.semantic_validator \
          --manifest data/manifests/p1_expanded_manifest.jsonl \
          --output reports/p1_semantic_validation.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from data_generation.oracle import DataValidationError, answer_from_edges
from validation.semantic_equivalence import (SemanticEquivalenceError,
                                             normalize_query,
                                             normalize_statement)

VALIDATOR_VERSION = "p1-0.2.0"
_QUESTION_LABELS = {"tr": "Soru: ", "en": "Question: "}
_ENTITY_LABELS = {"tr": "Varlıklar: ", "en": "Entities: "}
_CONDITIONS = {"canonical", "equivalent", "P1", "P2", "P3", "P4", "P5"}


def split_prompt(prompt: str, language: str) -> tuple[list[str], str]:
    label = _QUESTION_LABELS[language]
    if label not in prompt:
        raise ValueError("question label missing from prompt")
    head, _, tail = prompt.rpartition(label)
    premises = [line[2:].strip() for line in head.splitlines() if line.startswith("- ")]
    # The frozen prompt carries a trailing answer-format instruction line
    # after the question; the question itself is the first line of the tail.
    return premises, tail.splitlines()[0].strip()


def load_latents(root: Path, split: str) -> dict:
    latents = {}
    for line in (root / f"data/{split}/latent.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latents[row["query_instance_id"]] = row["record"]
    return latents


def surface_inventory(prompt: str, language: str) -> tuple[str, ...]:
    label = _ENTITY_LABELS[language]
    for line in prompt.splitlines():
        if line.startswith(label):
            return tuple(part.strip() for part in
                         line[len(label):].rstrip(".").split(","))
    raise ValueError("entity inventory line missing from prompt")


def check_row(row: dict, record: dict) -> dict:
    latent = tuple(record["latent_order"])
    result = {"prompt_id": row["prompt_id"], "language": row["language"],
              "condition": row["condition"], "relation_id": row["relation_id"],
              "status": "pass", "problems": []}
    if row["condition"] not in _CONDITIONS:
        result["problems"].append("unregistered_condition")
    try:
        premises, question = split_prompt(row["prompt"], row["language"])
        names = surface_inventory(row["prompt"], row["language"])
    except ValueError as error:
        result.update(status="structural_failure")
        result["problems"].append(str(error))
        return result
    # Name controls intentionally relabel entities; the surface inventory from
    # the prompt is authoritative. Only the native arm can be checked against
    # the latent chain edge-for-edge; relabeled arms are checked for exact
    # inventory size, strict-chain structure (via the oracle) and gold.
    native_arm = set(names) == set(latent)
    expected_edges = ({(latent[i], latent[i + 1]) for i in range(len(latent) - 1)}
                      if native_arm else None)
    if len(names) != len(latent) or len(set(names)) != len(names):
        result.update(status="structural_failure")
        result["problems"].append("inventory_size_or_uniqueness_mismatch")
        return result
    if len(premises) != len(names) - 1:
        result["problems"].append(
            f"premise_count_{len(premises)}_expected_{len(names) - 1}")
    asked_display = {row["display_query_subject"], row["display_query_object"]}
    if row["language"] == "en":
        # The closed normalization grammar is Turkish-only by design; English
        # rows receive structural checks (counts, entity references) and the
        # group-level gold-consistency check, recorded as such.
        missing = [name for name in names if name not in row["prompt"]]
        absent_query = [name for name in asked_display if name not in question]
        if missing:
            result["problems"].append("entities_missing_from_prompt")
        if absent_query:
            result["problems"].append("query_entities_missing_from_question")
        result["status"] = ("structural_failure" if result["problems"]
                           else "en_structural_only")
        return result
    edges, unparsed = set(), 0
    for premise in premises:
        try:
            edges.add(normalize_statement(premise, names, row["relation_id"]))
        except (SemanticEquivalenceError, ValueError):
            unparsed += 1
    if unparsed:
        result.update(status="not_normalizable")
        result["problems"].append(f"unparsed_premises_{unparsed}")
        return result
    if expected_edges is not None and edges != expected_edges:
        result.update(status="semantic_failure")
        result["problems"].append("edge_set_differs_from_latent_chain")
        return result
    try:
        subject, obj, relation = normalize_query(question, names, row["relation_id"])
    except (SemanticEquivalenceError, ValueError):
        result.update(status="not_normalizable")
        result["problems"].append("unparsed_query")
        return result
    if {subject, obj} != asked_display:
        result.update(status="semantic_failure")
        result["problems"].append("query_entities_differ")
        return result
    try:
        oracle = answer_from_edges(sorted(names), sorted(edges), subject, obj, relation)
    except (DataValidationError, TypeError, ValueError) as error:
        result.update(status="semantic_failure")
        result["problems"].append(f"oracle_error:{error}")
        return result
    if bool(oracle) != bool(row["gold_label"]):
        result.update(status="semantic_failure")
        result["problems"].append("oracle_answer_differs_from_gold")
        return result
    if result["problems"]:
        result["status"] = "structural_failure"
    result["semantic_digest"] = json.dumps(
        [sorted(edges), [subject, obj, relation]], ensure_ascii=False, sort_keys=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in
            (args.root / args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    latents = load_latents(args.root, rows[0]["split"])
    results, groups = [], collections.defaultdict(list)
    for row in rows:
        record = latents.get(row["query_instance_id"])
        if record is None:
            results.append({"prompt_id": row["prompt_id"], "status": "missing_latent_record"})
            continue
        outcome = check_row(row, record)
        results.append(outcome)
        groups[(row["query_instance_id"], row["relation_id"], row["name_control"])].append(
            (row["gold_label"], outcome.get("semantic_digest"), outcome["status"]))
    group_failures = 0
    for key, members in groups.items():
        golds = {g for g, _, _ in members}
        digests = {d for _, d, s in members if s == "pass" and d}
        if len(golds) != 1 or len(digests) > 1:
            group_failures += 1
    statuses = collections.Counter(r["status"] for r in results)
    by_condition = collections.defaultdict(collections.Counter)
    for row, outcome in zip(rows, results):
        by_condition[(row["language"], row["condition"])][outcome["status"]] += 1
    summary = {
        "validator_version": VALIDATOR_VERSION,
        "n_prompts": len(rows), "statuses": dict(statuses),
        "matched_groups": len(groups), "group_consistency_failures": group_failures,
        "by_language_condition": {f"{l}:{c}": dict(v) for (l, c), v in sorted(by_condition.items())},
        "scope_note": ("Automatic agreement with the implemented closed grammar; "
                       "not evidence of naturalness or human approval. "
                       "'not_normalizable' rows are the human-review residue."),
        "failures": [r for r in results if r["status"] not in
                     ("pass", "en_structural_only", "not_normalizable")][:200],
    }
    output = args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_prompts", "statuses",
                      "matched_groups", "group_consistency_failures")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
