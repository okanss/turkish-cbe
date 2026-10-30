"""Bounded, ordered process workers for complete-grid materialization.

Workers return only source recipes and measured SQLite tuples. The parent owns
all database writes and split leakage checks, in deterministic source order.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing
from pathlib import Path

from data_generation.generator import DEFAULT_DISTANCES, valid_distances
from experiments.dataset_builder import DatasetPlan, balanced_indices


MAX_WORKERS = 8
_TOKENIZERS = None


def _initialize(resource_dir: str) -> None:
    global _TOKENIZERS
    from .model_tokens import ModelTokenizers
    _TOKENIZERS = ModelTokenizers(Path(resource_dir))


def _jobs(plan: DatasetPlan):
    group_id, query_start = 0, 1
    for spec in plan.splits:
        for n in spec.sizes:
            query_count = 2 + len(valid_distances(n, DEFAULT_DISTANCES))
            for index in balanced_indices(spec.count, n):
                group_id += 1
                yield (group_id, query_start, asdict(spec), plan.presentation_seed, n, index)
                query_start += query_count


def _process(job, tokenizers=None) -> dict:
    # Import after process initialization; dataset_builder imports this module
    # to consume results and must not form an import-time dependency cycle.
    from data_generation.generator import GenerationPlan, generate_record
    from data_generation.validation import validate_record
    from .dataset_builder import (_component, _component_measurement, _design_rows,
        _json, _measurement, _queries, _rows_digest, _source_group, digest)
    from .grid_design import NAME_CONTROLS

    tokenizers = _TOKENIZERS if tokenizers is None else tokenizers
    if tokenizers is None:
        raise RuntimeError("worker tokenizers are not initialized")
    group_id, query_start, spec, presentation_seed, n, index = job
    generation_plan = GenerationPlan(seed=spec["seed"], sizes=tuple(spec["sizes"]),
        examples_per_size=4 * ((spec["count"] + 3) // 4), entity_types=("turkish_names",))
    endpoint = generate_record(generation_plan, n, index, "endpoint")
    validate_record(endpoint)
    source = _source_group(endpoint, presentation_seed)
    query_records, query_rows = [], []
    for query_id, record in enumerate(_queries(endpoint, generation_plan, n, index), query_start):
        validate_record(record)
        query_records.append((query_id, record))
        query_rows.append((query_id, group_id, record["instance_id"], record["query_type"],
            record["query_distance"], record["query_subject"], record["query_object"],
            record["query_relation"], record["query_direction"], int(record["gold_label"])))
    component_cache, component_rows, measurement_rows, logical_digests = {}, [], [], []
    for design in _design_rows():
        component = _component(source, design, component_cache)
        component_row, prepared, lengths, prefix_sha = _component_measurement(
            group_id, design[0], component, tokenizers)
        component_rows.append(component_row)
        rows = [_measurement(query_id, design[0], group_id, component, record,
                             tokenizers, prepared, lengths, prefix_sha)
                for query_id, record in query_records]
        measurement_rows.extend(rows)
        logical_digests.append(bytes.fromhex(_rows_digest(rows)))
    views = sorted({digest([source["name_maps"][control][language][name]
                    for name in endpoint["latent_order"]])
                    for control in NAME_CONTROLS for language in ("en", "tr")})
    return {
        "group_id": group_id, "split": spec["name"], "n": n, "index": index,
        "source": source, "ordering_id": endpoint["ordering_id"], "views": views,
        "group_row": (group_id, endpoint["ordering_id"], spec["name"], n, index, _json(source)),
        "query_rows": query_rows, "component_rows": component_rows,
        "measurement_rows": measurement_rows, "logical_digests": logical_digests,
    }


def iter_groups(plan: DatasetPlan, resource_dir: Path, workers: int = 1):
    """Yield measured groups in stable ID order with at most 2*workers jobs.

    The single-worker path is direct and supports stdin/interactive callers.
    Multiple workers use spawn for portable, isolated tokenizer initialization.
    Worker count affects execution only; source IDs and measurements agree.
    """
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer in [1, {MAX_WORKERS}]")
    if not isinstance(plan, DatasetPlan):
        raise ValueError("plan must be a DatasetPlan")
    plan.validate()
    resource_dir = Path(resource_dir).resolve()
    jobs = iter(_jobs(plan))
    if workers == 1:
        from .model_tokens import ModelTokenizers
        tokenizers = ModelTokenizers(resource_dir)
        for job in jobs:
            yield _process(job, tokenizers)
        return
    with ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize, initargs=(str(resource_dir),)) as executor:
        pending = deque()
        for _ in range(2 * workers):
            job = next(jobs, None)
            if job is None:
                break
            pending.append(executor.submit(_process, job))
        try:
            while pending:
                result = pending.popleft().result()
                job = next(jobs, None)
                if job is not None:
                    pending.append(executor.submit(_process, job))
                yield result
        finally:
            for future in pending:
                future.cancel()
