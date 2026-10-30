"""Reporting tests use temporary synthetic fixtures, never model-result files."""

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.reporting import export_figures, export_tables, _ci, _figure_jobs, _title


def profile(**changes):
    row = dict(model_id="model/example", model_family="fixture", model_version="revision-1",
               split="test", name_control="same_symbolic", relation_id="front",
               relation_family="spatial", query_type="distance_controlled", query_distance=1,
               n=4, language="tr", comparison="equivalent/P1", original_accuracy=.75,
               equivalent_accuracy=.5, invariance=.75, R=.5, D=.25, U=.25,
               strict_robust_correctness=.5, R_ci_low=.2, R_ci_high=.8,
               original_accuracy_ci_low=.4, original_accuracy_ci_high=.95)
    row.update(changes)
    return row


def difference(**changes):
    row = dict(model_id="model/example", model_family="fixture", split="test",
               name_control="same_symbolic", relation_id="front", relation_family="spatial",
               query_type="distance_controlled", query_distance=1, n=4,
               condition="canonical", difference=.25, ci_low=-.2, ci_high=.6)
    row.update(changes)
    return row


class AnalysisTableExportTests(unittest.TestCase):
    def test_csv_union_columns_order_missing_na_and_lossless_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = export_tables({"profiles": [{"label": "Ayşe, Özgür\nÇığ", "rate": .125},
                                                  {"label": "second", "undefined": None, "flag": False}]}, folder)
            self.assertEqual([path.name for path in paths], ["profiles.csv", "profiles.tex"])
            with paths[0].open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], ["label", "rate", "undefined", "flag"])
            self.assertEqual(rows[1], ["Ayşe, Özgür\nÇığ", "0.125", "NA", "NA"])
            self.assertEqual(rows[2], ["second", "NA", "NA", "false"])

    def test_latex_escapes_every_special_character_once(self):
        with tempfile.TemporaryDirectory() as folder:
            export_tables({"table": [{"a_b": "\\&%$#_{}~^<>\nÇığ"}]}, folder)
            text = (Path(folder) / "table.tex").read_text(encoding="utf-8")
            self.assertIn(r"a\_b", text)
            self.assertIn(r"\textbackslash{}\&\%\$\#\_\{\}\textasciitilde{}\textasciicircum{}\textless{}\textgreater{} Çığ", text)
            self.assertNotIn("\\\\textbackslash", text)
            self.assertIn(r"\begin{tabular}{l}", text)

    def test_empty_tables_and_empty_mapping_are_explicit_without_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(export_tables({}, folder), [])
            export_tables({"empty": []}, folder)
            self.assertEqual((Path(folder) / "empty.csv").read_bytes(), b"")
            self.assertIn("No observations", (Path(folder) / "empty.tex").read_text())

    def test_existing_output_blocks_all_writes_and_is_never_modified(self):
        with tempfile.TemporaryDirectory() as folder:
            existing = Path(folder) / "z.csv"
            existing.write_bytes(b"immutable")
            with self.assertRaises(FileExistsError):
                export_tables({"a": [{"value": 1}], "z": [{"value": 2}]}, folder)
            self.assertEqual(existing.read_bytes(), b"immutable")
            self.assertEqual(list(Path(folder).iterdir()), [existing])

    def test_filename_traversal_and_nonfinite_or_nested_cells_are_rejected_before_writes(self):
        cases = ({"../escape": []}, {"a/b": []}, {".hidden": []}, {"bad name": []},
                 {"ok": [{"v": float("nan")}]}, {"ok": [{"v": float("inf")}]},
                 {"ok": [{"v": []}]}, {"ok": [{None: 3}]}, {"ok": "rows"})
        for value in cases:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    export_tables(value, folder)
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_table_bytes_and_path_order_are_deterministic(self):
        with tempfile.TemporaryDirectory() as folder:
            first = export_tables({"z": [{"x": True}], "a": [{"y": "ı"}]}, Path(folder) / "first")
            second = export_tables({"a": [{"y": "ı"}], "z": [{"x": True}]}, Path(folder) / "second")
            self.assertEqual([path.name for path in first], ["a.csv", "a.tex", "z.csv", "z.tex"])
            self.assertEqual([path.read_bytes() for path in first], [path.read_bytes() for path in second])


class AnalysisFigureStructureTests(unittest.TestCase):
    def test_no_data_needs_no_matplotlib_import_or_output_directory(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", {"matplotlib": None}):
            destination = Path(folder) / "absent"
            self.assertEqual(export_figures({"profiles": [], "language_differences": []}, destination), [])
            self.assertFalse(destination.exists())

    def test_missing_optional_plot_dependency_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", {"matplotlib": None}):
            with self.assertRaisesRegex(RuntimeError, "optional"):
                export_figures({"profiles": [profile()]}, folder)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_all_six_families_exist_and_distance_keeps_n_separate(self):
        jobs = _figure_jobs({"profiles": [profile(), profile(n=8)],
                             "language_differences": [difference(), difference(n=8)]})
        self.assertEqual({job["number"] for job in jobs}, {1, 2, 3, 4, 5, 6})
        distance_jobs = [job for job in jobs if job["number"] == 6]
        self.assertEqual(len(distance_jobs), 2)
        self.assertEqual({dict(job["identity"])["n"] for job in distance_jobs}, {4, 8})
        self.assertEqual(len([job for job in jobs if job["number"] == 1]), 1)

    def test_figure_groups_never_pool_protected_or_unrecognized_dimensions(self):
        for field, other in (("model_id", "other-model"), ("model_version", "revision-2"),
                             ("split", "pilot"), ("name_control", "transliterated"),
                             ("relation_id", "behind"), ("relation_family", "temporal"),
                             ("query_type", "internal"), ("query_distance", 2),
                             ("comparison", "equivalent/P2"), ("new_factor", "extra")):
            rows = [profile(), profile(**{field: other})]
            jobs = _figure_jobs({"profiles": rows})
            self.assertEqual(len([job for job in jobs if job["number"] == 1]), 2, field)

    def test_language_series_stay_separate_and_rdu_has_language_facets(self):
        jobs = _figure_jobs({"profiles": [profile(language="tr"), profile(language="en")]})
        accuracy = [job for job in jobs if job["number"] == 1]
        self.assertEqual(len(accuracy), 1)
        self.assertEqual({row["language"] for row in accuracy[0]["rows"]}, {"en", "tr"})
        self.assertEqual(len([job for job in jobs if job["number"] == 4]), 2)

    def test_statistical_counts_and_intervals_do_not_become_experimental_factors(self):
        rows = [profile(n_clusters=10, n_invalid_original=2, org_only=1,
                        R_bootstrap_ci_low=.1, R_wilson_low=.15, R_wilson_high=.9),
                profile(n=8, n_clusters=12, n_invalid_original=3, org_only=2,
                        R_bootstrap_ci_low=.2, R_wilson_low=.25, R_wilson_high=.8)]
        jobs = _figure_jobs({"profiles": rows})
        accuracy = [job for job in jobs if job["number"] == 1]
        self.assertEqual(len(accuracy), 1)
        self.assertEqual(len(accuracy[0]["rows"]), 2)

    def test_endpoint_complexity_curves_validate_and_label_structural_distance(self):
        jobs = _figure_jobs({"profiles": [profile(n=4, query_type="endpoint", query_distance=3),
                                          profile(n=8, query_type="endpoint", query_distance=7)]})
        for number in (1, 2, 3, 4):
            matching = [job for job in jobs if job["number"] == number]
            self.assertEqual(len(matching), 1)
            self.assertEqual(len(matching[0]["rows"]), 2)
            self.assertNotIn("query_distance", dict(matching[0]["identity"]))
            self.assertIn("query distance = n − 1", _title(matching[0]))
        self.assertEqual(len([job for job in jobs if job["number"] == 6]), 2)
        differences = _figure_jobs({"language_differences": [difference(query_type="endpoint", query_distance=3),
                                                              difference(n=8, query_type="endpoint", query_distance=7)]})
        self.assertEqual(len(differences), 1)
        self.assertEqual(len(differences[0]["rows"]), 2)
        with self.assertRaisesRegex(ValueError, "Endpoint"):
            _figure_jobs({"profiles": [profile(n=8, query_type="endpoint", query_distance=1)]})

    def test_r_figure_prefers_the_explicit_wilson_interval(self):
        row = profile(R_wilson_low=.3, R_wilson_high=.7)
        self.assertEqual(_ci(row, "R", .5), (.3, .7))
        r_job = [job for job in _figure_jobs({"profiles": [row]}) if job["number"] == 3][0]
        self.assertIn("Confidence intervals: Wilson", _title(r_job))

    def test_bootstrap_status_and_interval_provenance_are_not_plot_dimensions(self):
        rows = [profile(bootstrap_R_status="defined", bootstrap_defined_replicates=2000,
                        R_wilson_status="independent", confidence_intervals='{"R":[0.2,0.8]}'),
                profile(n=8, bootstrap_R_status="partially_undefined", bootstrap_defined_replicates=1900,
                        R_wilson_status="descriptive", confidence_intervals='{"R":[0.3,0.9]}')]
        accuracy = [job for job in _figure_jobs({"profiles": rows}) if job["number"] == 1]
        self.assertEqual(len(accuracy), 1)
        self.assertEqual(len(accuracy[0]["rows"]), 2)

    def test_real_pipeline_language_curve_protects_design_not_test_diagnostics(self):
        from analysis.pipeline import build_tables
        from test_analysis_pipeline import cohort

        rows = cohort(n=4, clusters=2, prefix="reporting-regression")
        rows += cohort({("en", "canonical"): False, ("en", "equivalent"): False},
                       n=8, clusters=26, prefix="reporting-regression")
        tables = build_tables(rows, repeats=50)
        differences = tables["language_differences"]
        self.assertEqual({row["n"] for row in differences}, {4, 8})
        self.assertEqual({(row["n"], row["mcnemar_n"]) for row in differences}, {(4, 2), (8, 26)})
        self.assertEqual({row["method"] for row in differences}, {"exact", "asymptotic"})
        self.assertGreater(len({row["holm_family"] for row in differences}), 1)
        self.assertGreater(len({row["p_value"] for row in differences}), 1)
        jobs = [job for job in _figure_jobs(tables) if job["number"] == 5]
        self.assertEqual(len(jobs), 2)
        self.assertEqual({dict(job["identity"])["condition"] for job in jobs},
                         {"canonical", "equivalent"})
        for job in jobs:
            self.assertEqual({row["n"] for row in job["rows"]}, {4, 8})
            self.assertFalse({"b", "c", "method", "status", "statistic", "holm_family",
                              "holm_family_size", "mcnemar_n"} & dict(job["identity"]).keys())

    def test_duplicate_coordinates_require_an_explicit_prior_estimand(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _figure_jobs({"profiles": [profile(), profile(original_accuracy=.25)]})

    def test_invalid_rates_intervals_axes_or_rdu_are_rejected(self):
        cases = [profile(R=.8), profile(original_accuracy=1.1), profile(n=True),
                 profile(query_distance=0), profile(R_ci_low=.9, R_ci_high=.1),
                 profile(R_ci_low=None), profile(R=None), profile(original_accuracy=True)]
        for row in cases:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    _figure_jobs({"profiles": [row]})
        with self.assertRaises(ValueError):
            _figure_jobs({"language_differences": [difference(difference=1.01)]})

    def test_percentile_interval_need_not_contain_point_estimate(self):
        jobs = _figure_jobs({"profiles": [profile(R_ci_low=.6, R_ci_high=.8)]})
        self.assertTrue(jobs)

    def test_only_a_boolean_enables_synthetic_watermarks(self):
        with self.assertRaises(ValueError):
            export_figures({}, "unused", synthetic="yes")


@unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "optional matplotlib is not installed")
class AnalysisFigureExportTests(unittest.TestCase):
    def test_six_figures_export_three_formats_with_explicit_synthetic_watermarks(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = export_figures({"profiles": [profile()], "language_differences": [difference()]},
                                   folder, synthetic=True)
            self.assertEqual(len(paths), 18)
            self.assertEqual({path.suffix for path in paths}, {".pdf", ".svg", ".png"})
            self.assertEqual(len({path.stem for path in paths}), 6)
            for path in paths:
                self.assertGreater(path.stat().st_size, 100)
                if path.suffix == ".svg":
                    text = path.read_text(encoding="utf-8")
                    self.assertIn("SYNTHETIC TEST DATA", text)
                    self.assertIn("NOT MODEL RESULTS", text)
                    self.assertIn("<text", text)
                    self.assertNotIn("dc:date", text)
                elif path.suffix == ".png":
                    self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                    self.assertIn(b"SYNTHETIC TEST DATA", path.read_bytes())
                else:
                    self.assertTrue(path.read_bytes().startswith(b"%PDF"))
                    self.assertIn(b"SYNTHETIC TEST DATA", path.read_bytes())

    def test_figures_are_deterministic_and_refuse_to_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            data = {"language_differences": [difference(), difference(n=8, difference=.1)]}
            first = export_figures(data, Path(folder) / "first", synthetic=True)
            second = export_figures(data, Path(folder) / "second", synthetic=True)
            self.assertEqual([path.name for path in first], [path.name for path in second])
            self.assertEqual([path.read_bytes() for path in first], [path.read_bytes() for path in second])
            previous = [path.read_bytes() for path in first]
            with self.assertRaises(FileExistsError):
                export_figures(data, Path(folder) / "first", synthetic=True)
            self.assertEqual(previous, [path.read_bytes() for path in first])


if __name__ == "__main__":
    unittest.main()
