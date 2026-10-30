"""Synthetic local CoNLL-U fixtures; no corpus acquisition or inference occurs."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from analysis.lexical_bias.catalog import build_catalog
from analysis.lexical_bias.corpus import count_corpus


def word(identity, form, lemma=None, upos="NOUN"):
    lemma = form if lemma is None else lemma
    return "\t".join((str(identity), form, lemma, upos, "_", "_", "0", "root", "_", "_"))


def tiny_catalog(before="önce gelmiştir", after="sonra gelmiştir", before_lemma="önce", after_lemma="sonra"):
    return [{"relation_id": "synthetic", "family": "synthetic", "directions": [
        {"direction": direction, "term": term, "lemma": lemma, "lemma_tokens": lemma.split(),
         "surface_forms": [term]}
        for direction, term, lemma in (("before", before, before_lemma), ("after", after, after_lemma))]}]


def find(result, relation_id, direction="before"):
    return next(row for row in result["rows"] if row["relation_id"] == relation_id and row["direction"] == direction)


class LexicalCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "manifest.json"

    def manifest(self, texts, roles=None):
        roles = roles or ["train"] * len(texts)
        entries = []
        for i, (text, role) in enumerate(zip(texts, roles)):
            data = text.encode("utf-8")
            path = self.root / f"synthetic-{i}.conllu"
            path.write_bytes(data)
            entries.append({"path": path.name, "sha256": hashlib.sha256(data).hexdigest(),
                            "bytes": len(data), "role": role})
        metadata = {"schema_version": 1, "corpus_id": "synthetic-fixture-only", "version": "test-1",
            "revision": "synthetic-revision", "license": "synthetic-test-data",
            "license_url": "https://example.invalid/synthetic-license",
            "source_url": "https://example.invalid/synthetic-source", "files": entries}
        self.path.write_text(json.dumps(metadata), encoding="utf-8")
        return metadata

    def rewrite(self, metadata):
        self.path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_exact_surface_and_lemma_frequencies_use_separate_opportunities(self):
        text = "\n".join((word(1, "önce", "önce"), word(2, "gelmiştir", "gel"),
                          word(3, ".", ".", "PUNCT"))) + "\n\n"
        self.manifest([text])
        result = count_corpus(self.path, build_catalog())
        row = find(result, "temporal_arrival")
        self.assertEqual(row["surface_count"], 1)
        self.assertEqual(row["surface_opportunities"], 1)
        self.assertEqual(row["surface_frequency_per_million"], 1_000_000)
        self.assertEqual(row["surface_log10_per_million"], 6)
        self.assertEqual(row["lemma_count"], 1)
        self.assertEqual(row["lemma_opportunities"], 2)
        self.assertEqual(row["lemma_frequency_per_million"], 500_000)
        self.assertEqual(result["coverage"]["surface_tokens"], 2)
        self.assertEqual(result["coverage"]["syntactic_words"], 3)
        self.assertEqual(result["coverage"]["lemma_available"], 2)

    def test_multiword_surfaces_count_once_and_syntactic_lemmas_separately(self):
        text = "\n".join((word("1-2", "önündedir", "_", "_"),
                          word(1, "önünde", "ön"), word(2, "dir", "i", "AUX"),
                          word("2.1", "synthetic-empty", "synthetic-empty"),
                          word(3, ".", ".", "PUNCT")))
        self.manifest([text])
        result = count_corpus(self.path, build_catalog())
        row = find(result, "spatial_front")
        self.assertEqual(row["surface_count"], 1)
        self.assertEqual(row["surface_opportunities"], 1)
        self.assertEqual(row["lemma_count"], 1)
        self.assertEqual(row["lemma_opportunities"], 2)
        self.assertEqual(result["coverage"]["multiword_tokens"], 1)
        self.assertEqual(result["coverage"]["empty_nodes_skipped"], 1)
        self.assertEqual(result["coverage"]["surface_tokens"], 1)

    def test_turkish_i_mapping_and_decomposed_unicode_are_fixed(self):
        self.manifest(["\n".join((word(1, "I", "I"), word(2, "İ", "İ"),
                                   word(3, "O\u0308N", "O\u0308N")))])
        result = count_corpus(self.path, tiny_catalog("ı", "i", "ı", "i"))
        self.assertEqual(find(result, "synthetic", "before")["surface_count"], 1)
        self.assertEqual(find(result, "synthetic", "after")["surface_count"], 1)
        result = count_corpus(self.path, tiny_catalog("ön", "arka", "ön", "arka"))
        self.assertEqual(find(result, "synthetic")["surface_count"], 1)
        self.assertEqual(find(result, "synthetic")["lemma_count"], 1)

    def test_no_phrase_crosses_punctuation_sentence_or_source_file(self):
        one = "\n".join((word(1, "önce"), word(2, ",", ",", "PUNCT"), word(3, "gelmiştir", "gel")))
        two = word(1, "önce") + "\n\n" + word(1, "gelmiştir", "gel") + "\n\n" + word(1, "önce")
        three = word(1, "gelmiştir", "gel")
        self.manifest([one, two, three], ["train", "dev", "test"])
        result = count_corpus(self.path, tiny_catalog())
        self.assertEqual(find(result, "synthetic")["surface_count"], 0)
        self.assertEqual(result["coverage"]["sentences"], 5)
        self.assertEqual(result["coverage"]["roles"], {"train": 1, "dev": 1, "test": 1})

    def test_missing_lemma_is_barrier_and_coverage_is_not_imputed(self):
        self.manifest(["\n".join((word(1, "ön", "ön"), word(2, "bilinmeyen", "_"), word(3, "üst", "üst")))])
        catalog = tiny_catalog("ön bilinmeyen", "üst", "ön üst", "üst")
        result = count_corpus(self.path, catalog)
        row = find(result, "synthetic")
        self.assertEqual(row["lemma_count"], 0)
        self.assertEqual(row["lemma_opportunities"], 0)
        self.assertIsNone(row["lemma_frequency_per_million"])
        self.assertEqual(result["coverage"]["lemma_missing"], 1)
        self.assertEqual(result["coverage"]["lemma_available"], 2)
        self.assertAlmostEqual(result["coverage"]["lemma_coverage"], 2 / 3)

    def test_measured_zero_differs_from_missing_and_no_smoothing_is_used(self):
        self.manifest(["\n".join((word(1, "önce"), word(2, "koşmuştur", "koş")))])
        result = count_corpus(self.path, tiny_catalog())
        row = find(result, "synthetic")
        self.assertEqual(row["surface_count"], 0)
        self.assertEqual(row["surface_frequency_per_million"], 0)
        self.assertIsNone(row["surface_log10_per_million"])
        self.assertIsNone(row["direction_to_inverse_surface_ratio"])
        self.assertIsNone(row["direction_to_inverse_lemma_ratio"])
        missing = count_corpus(None, tiny_catalog())
        self.assertIsNone(find(missing, "synthetic")["surface_count"])
        self.assertIsNone(find(missing, "synthetic")["surface_frequency_per_million"])
        self.assertEqual(missing["coverage"]["status"], "missing_corpus")

    def test_inverse_ratio_uses_explicit_frequencies_and_never_invents_denominator(self):
        text = "\n".join((word(1, "önce"), word(2, "önce"), word(3, "sonra")))
        self.manifest([text])
        result = count_corpus(self.path, tiny_catalog("önce", "sonra"))
        self.assertEqual(find(result, "synthetic")["direction_to_inverse_surface_ratio"], 2)
        self.assertEqual(find(result, "synthetic", "after")["direction_to_inverse_lemma_ratio"], .5)

    def test_missing_manifest_produces_all_null_frequencies(self):
        catalog = build_catalog()
        for path in (None, self.root / "absent.json"):
            result = count_corpus(path, catalog)
            self.assertEqual(len(result["rows"]), 18)
            self.assertEqual(result["input_files_sha256"], {})
            for row in result["rows"]:
                self.assertEqual(row["status"], "missing_corpus")
                self.assertIsNone(row["surface_count"])
                self.assertIsNone(row["lemma_count"])

    def test_all_declared_files_are_hash_verified_and_input_is_not_mutated(self):
        metadata = self.manifest([word(1, "önce"), word(1, "sonra")], ["train", "test"])
        catalog = tiny_catalog("önce", "sonra")
        before = copy.deepcopy(catalog)
        result = count_corpus(self.path, catalog)
        self.assertEqual(catalog, before)
        self.assertEqual(result, count_corpus(self.path, catalog))
        self.assertEqual(len(result["input_files_sha256"]), 3)
        for entry in metadata["files"]:
            self.assertEqual(result["input_files_sha256"][entry["path"]], entry["sha256"])
        self.assertEqual(result["corpus"]["revision"], "synthetic-revision")

    def test_actual_surface_variants_are_counted_separately(self):
        catalog = tiny_catalog("önündedir", "arkasındadır", "ön", "arka")
        catalog[0]["directions"][0]["surface_forms"].append("önünde midir")
        self.manifest(["\n".join((word(1, "önünde", "ön"), word(2, "midir", "mi")))])
        row = find(count_corpus(self.path, catalog), "synthetic")
        self.assertEqual(row["surface_count"], 0)
        self.assertEqual(row["surface_form_counts"][0]["count"], 0)
        self.assertEqual(row["surface_form_counts"][1]["count"], 1)

    def test_changed_hash_size_or_missing_declared_file_reject_partial_counts(self):
        for mutation in ("hash", "size", "missing"):
            with self.subTest(mutation=mutation):
                metadata = self.manifest([word(1, "önce")])
                if mutation == "hash":
                    metadata["files"][0]["sha256"] = "0" * 64
                elif mutation == "size":
                    metadata["files"][0]["bytes"] += 1
                else:
                    (self.root / metadata["files"][0]["path"]).unlink()
                self.rewrite(metadata)
                with self.assertRaises(ValueError):
                    count_corpus(self.path, tiny_catalog())

    def test_unsafe_paths_duplicates_and_symlinks_are_rejected(self):
        for path in ("../outside.conllu", "/absolute.conllu", "a/../b.conllu", "./synthetic-0.conllu"):
            metadata = self.manifest([word(1, "önce")])
            metadata["files"][0]["path"] = path
            self.rewrite(metadata)
            with self.subTest(path=path), self.assertRaises(ValueError):
                count_corpus(self.path, tiny_catalog())
        metadata = self.manifest([word(1, "önce")])
        metadata["files"].append(dict(metadata["files"][0]))
        self.rewrite(metadata)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            count_corpus(self.path, tiny_catalog())
        metadata = self.manifest([word(1, "önce")])
        original = self.root / metadata["files"][0]["path"]
        link = self.root / "link.conllu"
        link.symlink_to(original)
        metadata["files"][0]["path"] = link.name
        self.rewrite(metadata)
        with self.assertRaisesRegex(ValueError, "symlink"):
            count_corpus(self.path, tiny_catalog())

    def test_manifest_schema_provenance_duplicate_json_and_nonfinite_rejected(self):
        for key in ("schema_version", "corpus_id", "version", "revision", "license", "license_url", "source_url"):
            metadata = self.manifest([word(1, "önce")])
            del metadata[key]
            self.rewrite(metadata)
            with self.subTest(key=key), self.assertRaises(ValueError):
                count_corpus(self.path, tiny_catalog())
        for text in ('{"schema_version":1,"schema_version":1}', '{"schema_version":NaN}'):
            self.path.write_text(text)
            with self.assertRaises(ValueError):
                count_corpus(self.path, tiny_catalog())

    def test_malformed_conllu_id_range_column_and_whitespace_rejected(self):
        cases = ["1\ttoo-few-columns", word("bad", "önce"), word(2, "önce"),
                 word(1, "önce") + "\n" + word(1, "sonra"),
                 word("1-3", "synthetic") + "\n" + word(1, "önce") + "\n" + word(2, "sonra"),
                 word("1-2", "synthetic") + "\n" + word("2-3", "synthetic") + "\n"
                 + "\n".join(word(i, "önce") for i in (1, 2, 3)),
                 word(1, "two words"), word(1, "önce", "two lemmas")]
        for text in cases:
            self.manifest([text])
            with self.subTest(text=text), self.assertRaises(ValueError):
                count_corpus(self.path, tiny_catalog())

    def test_empty_measured_corpus_has_zero_counts_and_undefined_rates(self):
        self.manifest(["# synthetic empty fixture\n"])
        result = count_corpus(self.path, tiny_catalog())
        row = find(result, "synthetic")
        self.assertEqual(row["surface_count"], 0)
        self.assertEqual(row["surface_opportunities"], 0)
        self.assertIsNone(row["surface_frequency_per_million"])
        self.assertEqual(result["coverage"]["status"], "counted")
        self.assertEqual(result["coverage"]["surface_tokens"], 0)
        self.assertIsNone(result["coverage"]["lemma_coverage"])

    def test_comments_and_punctuation_never_count_as_lexical_text(self):
        text = "# text = önce gelmiştir\n" + word(1, ".", ".", "PUNCT")
        self.manifest([text])
        result = count_corpus(self.path, tiny_catalog())
        self.assertEqual(find(result, "synthetic")["surface_count"], 0)
        self.assertEqual(result["coverage"]["surface_tokens"], 0)
        self.assertEqual(result["coverage"]["syntactic_punctuation"], 1)

    def test_retained_license_and_readme_are_verified_but_never_parsed_as_corpus(self):
        metadata = self.manifest([word(1, "önce"), "Synthetic attribution text.",
                                  "Synthetic fixture license."], ["train", "source_readme", "license"])
        metadata["attribution"] = "Synthetic authors only."
        self.rewrite(metadata)
        result = count_corpus(self.path, tiny_catalog("önce", "sonra"))
        self.assertEqual(find(result, "synthetic")["surface_count"], 1)
        self.assertEqual(result["coverage"]["files"], 3)
        self.assertEqual(result["coverage"]["corpus_files"], 1)
        self.assertEqual(result["coverage"]["metadata_files"], 2)
        self.assertEqual(result["coverage"]["sentences"], 1)
        self.assertEqual(result["corpus"]["attribution"], "Synthetic authors only.")
        self.assertEqual(len(result["metadata_files"]), 2)
        (self.root / metadata["files"][1]["path"]).write_text("Changed attribution.")
        with self.assertRaisesRegex(ValueError, "hash or byte"):
            count_corpus(self.path, tiny_catalog())

    def test_missing_corpus_files_and_unregistered_roles_are_not_zero_evidence(self):
        for role in ("source_readme", "unregistered"):
            self.manifest(["Synthetic metadata text."], [role])
            with self.subTest(role=role), self.assertRaises(ValueError):
                count_corpus(self.path, tiny_catalog())

    def test_boolean_schema_version_is_rejected(self):
        metadata = self.manifest([word(1, "önce")])
        metadata["schema_version"] = True
        self.rewrite(metadata)
        with self.assertRaisesRegex(ValueError, "schema"):
            count_corpus(self.path, tiny_catalog())


if __name__ == "__main__":
    unittest.main()
