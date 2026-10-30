"""Dataset-bound two-reviewer evidence for large evaluations.

This validates recorded human decisions and their sampling scope. It cannot
independently attest reviewer identity or certify every unseen Turkish string.
No approval is inferred from an LLM, a parser round trip or an 'approved' flag.
"""

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

from .dataset import GridDataset
from complete_grid import dataset_builder as grid

LARGE_EVALUATION_CALLS = 2000
MINIMUM_REVIEW_SAMPLES = 100
MINIMUM_PER_TEMPLATE_CELL = 20
AXES = ("relation_id", "condition", "name_control", "query_type", "n")
RATINGS = tuple(f"reviewer{i}_{axis}" for i in (1, 2)
                for axis in ("grammar", "naturalness", "equivalence", "morphology"))


def _read(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("linguistic review evidence must not traverse symlinks")
    return path.read_bytes()


def required_axes(selections):
    aliases = {"relation_id":"relations", "condition":"conditions", "name_control":"name_controls",
               "query_type":"query_types", "n":"sizes"}
    return {axis:sorted({v for selection in selections for v in selection[aliases[axis]]}) for axis in AXES}


def validate_review(root, manifest, review_path, selections, *, samples_path=None, allow_fixture=False):
    """Validate exact sample prompts, two human ratings and operational scope.

    A template cell is relation × presentation × name control; at least 20
    distinct prompts per required cell, at least 100 overall, and coverage of
    every requested n/query-type are required. This is sampled human review,
    not an exhaustive naturalness proof. Selections must name actual cohorts.
    """
    if review_path is None:
        raise ValueError("approved two-reviewer linguistic review is required before large evaluation")
    root = Path(root).resolve()
    review_path = Path(review_path)
    if not review_path.is_absolute(): review_path = root / review_path
    raw = _read(review_path)
    # Reject duplicate keys and NaN, as in native result readers, without an
    # analysis import cycle during runner initialization.
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result: raise ValueError("duplicate linguistic review JSON key")
            result[key] = value
        return result
    review = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite review JSON")))
    if (review.get("schema_version") != 1 or review.get("artifact_type") != "turkish_linguistic_review"
            or review.get("status") != "approved"):
        raise ValueError("versioned approved linguistic review receipt required")
    if review.get("fixture_only") is True and not allow_fixture:
        raise ValueError("synthetic review fixtures cannot authorize live evaluation")
    reviewers = review.get("reviewers")
    if (not isinstance(reviewers, list) or len(reviewers) != 2
            or any(not isinstance(r,str) or not r.strip() for r in reviewers)
            or len(set(reviewers)) != 2):
        raise ValueError("two identified independent Turkish-proficient reviewers are required")
    if review.get("reviewer_qualification") != "two_independent_turkish_proficient_reviewers":
        raise ValueError("reviewer qualifications must be explicitly recorded")
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute(): manifest_path = root / manifest_path
    manifest_sha = hashlib.sha256(_read(manifest_path)).hexdigest()
    if review.get("dataset_manifest_sha256") != manifest_sha:
        raise ValueError("linguistic review is bound to another frozen dataset")
    entry = review.get("samples")
    if not isinstance(entry,dict) or not isinstance(entry.get("path"),str):
        raise ValueError("hash-bound reviewed sample CSV required")
    sample_file = Path(samples_path) if samples_path is not None else review_path.parent / entry["path"]
    if not sample_file.is_absolute(): sample_file = root / sample_file
    sample_bytes = _read(sample_file)
    if hashlib.sha256(sample_bytes).hexdigest() != entry.get("sha256"):
        raise ValueError("reviewed sample CSV hash changed")
    reader = csv.DictReader(io.StringIO(sample_bytes.decode("utf-8"), newline=""))
    required = {"query_id","design_id","prompt_sha256","prompt",*AXES,*RATINGS,"resolution"}
    if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames) or not required <= set(reader.fieldnames):
        raise ValueError("review CSV is missing required sample identity or two-reviewer decisions")
    rows = list(reader)
    if len(rows) < MINIMUM_REVIEW_SAMPLES or entry.get("rows") != len(rows):
        raise ValueError("at least 100 distinct reviewed sample prompts are required")
    seen, coverage, cells = set(), {axis:set() for axis in AXES}, Counter()
    with GridDataset(root, manifest) as dataset:
        flags = dataset.manifest["validation"]
        if not flags.get("actual_surface_normal_forms") or not flags.get("independent_oracles_agree"):
            raise ValueError("automated surface-equivalence and independent-oracle checks must pass")
        tokenizers = grid.ModelTokenizers(root / "tokenizers")
        for row in rows:
            if any(row.get(key) != "pass" for key in RATINGS) or row.get("resolution") != "approved":
                raise ValueError("pending, rejected or unresolved human decisions cannot authorize evaluation")
            try: key = (int(row["query_id"]), int(row["design_id"]))
            except (ValueError,TypeError): raise ValueError("review sample IDs must be exact grid IDs") from None
            if key in seen: raise ValueError("duplicate reviewed sample prompt")
            seen.add(key)
            sample = grid._materialized(root,dataset.connection,*key,tokenizers)
            if sample["language"] != "tr" or sample["prompt_sha256"] != row["prompt_sha256"] or sample["prompt"] != row["prompt"]:
                raise ValueError("review sample text is not the identified frozen Turkish prompt")
            for axis in AXES:
                if str(sample[axis]) != row[axis]: raise ValueError("review sample coverage metadata changed")
                coverage[axis].add(sample[axis])
            cells[(sample["relation_id"],sample["condition"],sample["name_control"])] += 1
    needed = required_axes(selections)
    if any(not set(values) <= coverage[axis] for axis,values in needed.items()):
        raise ValueError("linguistic review does not cover the evaluation n/query/template axes")
    for relation in needed["relation_id"]:
        for condition in needed["condition"]:
            for control in needed["name_control"]:
                if cells[relation,condition,control] < MINIMUM_PER_TEMPLATE_CELL:
                    raise ValueError("each required relation/presentation/name template cell needs at least 20 reviewed prompts")
    return {"schema_version":1, "dataset_manifest_sha256":manifest_sha,
        "receipt_sha256":hashlib.sha256(raw).hexdigest(), "samples_sha256":hashlib.sha256(sample_bytes).hexdigest(),
        "review_path":str(review_path.absolute()), "samples_path":str(sample_file.absolute()),
        "reviewers":reviewers,"sample_count":len(rows),"required_axes":needed,
        "observed_axes":{k:sorted(v) for k,v in coverage.items()},
        "coverage_scope":"sampled template cells; not exhaustive human certification or independent identity attestation"}
