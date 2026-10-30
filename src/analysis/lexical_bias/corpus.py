"""Hash-verified, offline exact surface/lemma counts for pinned CoNLL-U files."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import unicodedata

from renderers.morphology import turkish_lower


def _normal(text):
    return turkish_lower(unicodedata.normalize("NFC", text))


def _phrase(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("catalog phrases must be nonempty strings")
    return tuple(_normal(word) for word in text.split())


def _catalog(catalog):
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("catalog must be a nonempty list")
    records, seen = [], set()
    for relation in catalog:
        if not isinstance(relation, dict) or not isinstance(relation.get("directions"), list):
            raise ValueError("invalid relation catalog")
        relation_id, family = relation.get("relation_id"), relation.get("family")
        if not isinstance(relation_id, str) or not relation_id or not isinstance(family, str) or not family:
            raise ValueError("catalog relation identity is missing")
        if (any(not isinstance(item, dict) for item in relation["directions"])
                or {item.get("direction") for item in relation["directions"]} != {"before", "after"}):
            raise ValueError("each catalog relation needs both directions")
        for item in relation["directions"]:
            identity = (relation_id, item["direction"])
            if identity in seen:
                raise ValueError("duplicate relation direction")
            seen.add(identity)
            term = _phrase(item.get("term"))
            lemma_tokens = item.get("lemma_tokens")
            if not isinstance(lemma_tokens, list) or not lemma_tokens or any(
                    not isinstance(word, str) or not word.strip() or len(word.split()) != 1 for word in lemma_tokens):
                raise ValueError("lemma_tokens must contain individual nonempty words")
            forms = item.get("surface_forms")
            if not isinstance(forms, list) or not forms:
                raise ValueError("surface_forms must be nonempty")
            patterns = list(dict.fromkeys(_phrase(form) for form in forms))
            if term not in patterns:
                raise ValueError("surface_forms must include the canonical term")
            records.append({"relation_id": relation_id, "family": family,
                "direction": item["direction"], "lemma": item["lemma"], "term": item["term"],
                "surface_pattern": term, "lemma_pattern": tuple(_normal(word) for word in lemma_tokens),
                "surface_forms": [(form, _phrase(form)) for form in dict.fromkeys(forms)]})
    return records


def _decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate corpus manifest key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite manifest JSON")))


def _punctuation(form, upos):
    return upos == "PUNCT" or all(unicodedata.category(char)[0] in {"P", "S"} for char in form)


def _sentences(text):
    """Yield CoNLL-U sentences; comments and empty nodes never become text."""
    current = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            if current:
                yield current
                current = []
            continue
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 10 or not fields[1]:
            raise ValueError(f"invalid CoNLL-U row at line {number}")
        current.append(fields)
    if current:
        yield current


def _layers(sentence, coverage):
    words, ranges, empty = {}, {}, 0
    order = []
    for fields in sentence:
        identity = fields[0]
        if re.fullmatch(r"[1-9][0-9]*", identity):
            position = int(identity)
            if position in words:
                raise ValueError("duplicate CoNLL-U word ID")
            words[position] = fields
            order.append(position)
        elif re.fullmatch(r"[1-9][0-9]*-[1-9][0-9]*", identity):
            first, last = map(int, identity.split("-"))
            if first >= last or first in ranges:
                raise ValueError("invalid or duplicate CoNLL-U multiword range")
            ranges[first] = (last, fields)
        elif re.fullmatch(r"[1-9][0-9]*\.[1-9][0-9]*", identity):
            empty += 1
        else:
            raise ValueError("invalid CoNLL-U token ID")
    if not words or order != list(range(1, len(words) + 1)):
        raise ValueError("CoNLL-U syntactic word IDs must be contiguous and ordered")
    covered = set()
    for first, (last, _) in ranges.items():
        positions = set(range(first, last + 1))
        if not positions <= words.keys() or positions & covered:
            raise ValueError("overlapping or incomplete CoNLL-U multiword range")
        covered.update(positions)
    surface, lemmas = [], []
    for position in order:
        fields = words[position]
        form, lemma, upos = fields[1], fields[2], fields[3]
        if any(char.isspace() for char in form):
            raise ValueError("CoNLL-U FORM containing whitespace has no registered counting rule")
        punctuation = _punctuation(form, upos)
        coverage["syntactic_words"] += 1
        if punctuation:
            lemmas.append(None)
            coverage["syntactic_punctuation"] += 1
        elif lemma == "_":
            lemmas.append(None)
            coverage["lemma_missing"] += 1
        else:
            if not lemma or any(char.isspace() for char in lemma):
                raise ValueError("CoNLL-U LEMMA must be one word or underscore")
            lemmas.append(_normal(lemma))
            coverage["lemma_available"] += 1
    position = 1
    while position <= len(words):
        if position in ranges:
            last, fields = ranges[position]
            form = fields[1]
            if any(char.isspace() for char in form):
                raise ValueError("multiword surface FORM must be one orthographic token")
            punctuation = all(_punctuation(words[i][1], words[i][3]) for i in range(position, last + 1))
            surface.append(None if punctuation or _punctuation(form, fields[3]) else _normal(form))
            coverage["multiword_tokens"] += 1
            position = last + 1
        else:
            fields = words[position]
            surface.append(None if _punctuation(fields[1], fields[3]) else _normal(fields[1]))
            position += 1
    coverage["sentences"] += 1
    coverage["empty_nodes_skipped"] += empty
    coverage["surface_tokens"] += sum(token is not None for token in surface)
    coverage["surface_punctuation"] += sum(token is None for token in surface)
    return surface, lemmas


def _accumulate(tokens, patterns, counts, opportunities):
    for length, registered in patterns.items():
        for start in range(len(tokens) - length + 1):
            candidate = tuple(tokens[start:start + length])
            if None in candidate:
                continue
            opportunities[length] += 1
            if candidate in registered:
                counts[candidate] += 1


def _frequency(count, opportunities):
    rate = count * 1_000_000 / opportunities if opportunities else None
    return rate, math.log10(rate) if rate is not None and rate > 0 else None


def _missing(records):
    rows = []
    for record in records:
        row = {key: record[key] for key in ("relation_id", "family", "direction", "lemma", "term")}
        for layer in ("surface", "lemma"):
            for suffix in ("count", "opportunities", "frequency_per_million", "log10_per_million"):
                row[layer + "_" + suffix] = None
        row.update(status="missing_corpus", surface_form_counts=[],
                   direction_to_inverse_surface_ratio=None, direction_to_inverse_lemma_ratio=None)
        rows.append(row)
    return {"rows": rows, "coverage": {"status": "missing_corpus", "files": 0,
            "surface_tokens": None, "syntactic_words": None, "lemma_available": None,
            "lemma_missing": None, "sentences": None}, "input_files_sha256": {},
            "normalization": "Unicode NFC; Turkish I→ı and İ→i; exact case-insensitive ngrams",
            "frequency_measure": "per_million_eligible_ngram_opportunities", "corpus": None}


def count_corpus(manifest_path, catalog) -> dict:
    """Count actual, verified corpus observations; never fetch or infer lemmas.

    Missing manifest/None returns null frequencies.  A present manifest with a
    missing, changed or malformed declared file raises, preventing silent partial
    corpus results.  Surface and lemma counts have separate ngram opportunity
    denominators. Punctuation, unknown lemmas and sentence boundaries are barriers.
    Zero measured counts retain zero frequency when opportunities exist; log10(0)
    and inverse-zero ratios remain None without smoothing.
    """
    records = _catalog(catalog)
    if manifest_path is None or not Path(manifest_path).exists():
        return _missing(records)
    path = Path(manifest_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("corpus manifest must be an owned regular file")
    path = path.resolve()
    data = path.read_bytes()
    manifest = _decode(data)
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != 1):
        raise ValueError("unsupported corpus manifest schema")
    for key in ("corpus_id", "version", "revision", "license", "license_url", "source_url"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError("missing corpus provenance: " + key)
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("corpus manifest needs declared source files")
    hashes = {"manifest:" + path.name: hashlib.sha256(data).hexdigest()}
    verified, seen, file_roles = [], set(), Counter()
    metadata_files = []
    for entry in entries:
        if not isinstance(entry, dict) or not {"path", "sha256", "bytes", "role"} <= entry.keys():
            raise ValueError("invalid corpus file entry")
        relative = entry["path"]
        if not isinstance(relative, str) or not relative or relative in seen:
            raise ValueError("duplicate or invalid corpus file path")
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or parsed.as_posix() != relative or any(part in {".", ".."} for part in parsed.parts):
            raise ValueError("corpus file escapes its manifest directory")
        seen.add(relative)
        if (type(entry["bytes"]) is not int or entry["bytes"] < 0
                or not isinstance(entry["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
                or not isinstance(entry["role"], str) or not entry["role"]):
            raise ValueError("invalid corpus size/hash/role")
        source = path.parent
        for component in parsed.parts:
            source /= component
            if source.is_symlink():
                raise ValueError("corpus source symlinks are not allowed")
        if not source.is_file() or not source.resolve().is_relative_to(path.parent):
            raise ValueError("declared corpus source file is missing")
        raw = source.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if len(raw) != entry["bytes"] or sha != entry["sha256"]:
            raise ValueError("corpus source hash or byte length mismatch")
        hashes[relative] = sha
        role = entry["role"]
        file_roles[role] += 1
        if role in {"train", "dev", "test"}:
            verified.append((role, raw.decode("utf-8")))
        elif role in {"license", "source_readme", "readme", "metadata"}:
            metadata_files.append({"path": relative, "role": role, "sha256": sha})
        else:
            raise ValueError("unregistered corpus file role: " + role)
    if not verified:
        raise ValueError("corpus manifest has no declared CoNLL-U train/dev/test files")
    patterns = {layer: {} for layer in ("surface", "lemma")}
    for record in records:
        for layer, registered in (("surface", [p for _, p in record["surface_forms"]]),
                                  ("lemma", [record["lemma_pattern"]])):
            for pattern in registered:
                patterns[layer].setdefault(len(pattern), set()).add(pattern)
    counts = {"surface": Counter(), "lemma": Counter()}
    opportunities = {"surface": Counter(), "lemma": Counter()}
    coverage = {key: 0 for key in ("sentences", "syntactic_words", "syntactic_punctuation",
        "surface_tokens", "surface_punctuation", "lemma_available", "lemma_missing",
        "multiword_tokens", "empty_nodes_skipped")}
    for _, text in verified:
        for sentence in _sentences(text):
            surface, lemmas = _layers(sentence, coverage)
            for layer, tokens in (("surface", surface), ("lemma", lemmas)):
                _accumulate(tokens, patterns[layer], counts[layer], opportunities[layer])
    rows = []
    for record in records:
        row = {key: record[key] for key in ("relation_id", "family", "direction", "lemma", "term")}
        for layer in ("surface", "lemma"):
            pattern = record[layer + "_pattern"]
            count, denominator = counts[layer][pattern], opportunities[layer][len(pattern)]
            rate, log = _frequency(count, denominator)
            row.update({layer + "_count": count, layer + "_opportunities": denominator,
                        layer + "_frequency_per_million": rate, layer + "_log10_per_million": log})
        forms = []
        for form, pattern in record["surface_forms"]:
            count, denominator = counts["surface"][pattern], opportunities["surface"][len(pattern)]
            rate, log = _frequency(count, denominator)
            forms.append({"form": form, "count": count, "opportunities": denominator,
                          "frequency_per_million": rate, "log10_per_million": log})
        row.update(surface_form_counts=forms, status="counted")
        rows.append(row)
    by_id = {(row["relation_id"], row["direction"]): row for row in rows}
    for row in rows:
        inverse = by_id[(row["relation_id"], "after" if row["direction"] == "before" else "before")]
        for layer in ("surface", "lemma"):
            numerator = row[layer + "_frequency_per_million"]
            denominator = inverse[layer + "_frequency_per_million"]
            row["direction_to_inverse_" + layer + "_ratio"] = (
                numerator / denominator if numerator is not None and denominator is not None and denominator > 0 else None)
    coverage.update(status="counted", files=len(entries), corpus_files=len(verified),
        metadata_files=len(metadata_files), roles=dict(file_roles),
        counted_roles=dict(Counter(role for role, _ in verified)),
        lemma_coverage=(coverage["lemma_available"] / (coverage["lemma_available"] + coverage["lemma_missing"])
                        if coverage["lemma_available"] + coverage["lemma_missing"] else None),
        surface_ngram_opportunities={str(k):v for k,v in sorted(opportunities["surface"].items())},
        lemma_ngram_opportunities={str(k):v for k,v in sorted(opportunities["lemma"].items())})
    return {"rows": rows, "coverage": coverage, "input_files_sha256": hashes,
        "normalization": "Unicode NFC; Turkish I→ı and İ→i; exact case-insensitive ngrams",
        "frequency_measure": "per_million_eligible_ngram_opportunities",
        "zero_policy": "No smoothing; measured zero counts retained; log10(0) and inverse-zero ratios null",
        "metadata_files": metadata_files,
        "corpus": {key: manifest[key] for key in ("corpus_id", "version", "revision", "license", "license_url", "source_url",
            "attribution", "reference_url", "scope") if key in manifest}}
