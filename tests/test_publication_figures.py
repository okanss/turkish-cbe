"""Publication graphics tests use temporary synthetic fixtures only."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from publication.figures import PLACEHOLDERS, _bias_jobs, export_paper_figures
from publication.tables import build_paper_tables


def evidence(**changes):
    row={"contains_model_results":False,"synthetic_test_fixture":True,"dataset_rows":[],
        "dataset_metadata":{},"conditions":[],"inference_tables":{},"lexical_tables":{}}
    row.update(changes)
    return row


def profile(**changes):
    row={"model_id":"fixture/model","model_family":"fixture","model_revision":"revision-1",
        "model_spec_sha256":"a"*64,"split":"test","name_control":"symbolic_shared",
        "relation_id":"front","relation_family":"spatial","query_type":"distance_controlled",
        "query_distance":1,"n":4,"language":"tr","comparison":"equivalent",
        "original_accuracy":.75,"equivalent_accuracy":.5,"invariance":.75,"R":.5,"D":.25,"U":.25,
        "R_wilson_low":.2,"R_wilson_high":.8,"n_pairs":8,"n_valid_pairs":8,"n_clusters":8}
    row.update(changes)
    return row


class PublicationFigureLogicTests(unittest.TestCase):
    def test_bias_uses_native_q_and_valid_denominator_separately_from_gold(self):
        rows=[{"model_id":"fixture","query_relation":"before","q":.75,"n_valid":8,"gold_yes_planned":5},
              {"model_id":"fixture","query_relation":"after","q":.25,"n_valid":4,"gold_yes_planned":3}]
        jobs=_bias_jobs(rows)
        self.assertEqual(len(jobs),2)
        self.assertEqual([(job[2],job[3]) for job in jobs],[(.75,8),(.25,4)])

    def test_bias_requires_valid_sample_and_valid_rates(self):
        self.assertEqual(_bias_jobs([{"q":.5,"n_valid":0},{"q":None,"n_valid":10}]),[])
        for value in (float("nan"),-1,2,True,"0.5"):
            with self.subTest(value=value),self.assertRaises(ValueError):_bias_jobs([{"q":value,"n_valid":1}])

    def test_duplicate_bias_cells_are_rejected(self):
        row={"q":.5,"n_valid":2,"model_id":"fixture","query_relation":"before"}
        with self.assertRaises(ValueError):_bias_jobs([row,row.copy()])

    def test_synthetic_fixture_requires_explicit_export_switch(self):
        fixture=evidence()
        with tempfile.TemporaryDirectory() as folder,self.assertRaisesRegex(ValueError,"Synthetic"):
            export_paper_figures(fixture,build_paper_tables(fixture),folder)

    def test_missing_plotting_dependency_is_explicit(self):
        fixture=evidence()
        with tempfile.TemporaryDirectory() as folder,patch.dict("sys.modules",{"matplotlib":None}):
            with self.assertRaisesRegex(RuntimeError,"plotting"):
                export_paper_figures(fixture,build_paper_tables(fixture),folder,synthetic=True)
            self.assertEqual(list(Path(folder).iterdir()),[])


@unittest.skipUnless(importlib.util.find_spec("matplotlib"),"Pinned plotting runtime required")
class PublicationFigureExportTests(unittest.TestCase):
    def test_diagrams_all_formats_and_four_missing_empirical_placeholders(self):
        fixture=evidence()
        with tempfile.TemporaryDirectory() as folder:
            paths=export_paper_figures(fixture,build_paper_tables(fixture),folder,synthetic=True)
            self.assertEqual(len(paths),10)
            self.assertEqual(sum(path.suffix==".md" for path in paths),4)
            self.assertEqual(sum(path.suffix==".pdf" for path in paths),2)
            for path in paths:
                self.assertGreater(path.stat().st_size,100)
                if path.suffix==".md":self.assertIn("NO VERIFIED",path.read_text())
            svg=(Path(folder)/"figure_1_architecture.svg").read_text()
            self.assertIn("Language-independent",svg)
            self.assertIn("not experimental results",svg)

    def test_diagram_bytes_are_deterministic(self):
        fixture=evidence()
        with tempfile.TemporaryDirectory() as folder:
            first=export_paper_figures(fixture,build_paper_tables(fixture),Path(folder)/"one",synthetic=True)
            second=export_paper_figures(fixture,build_paper_tables(fixture),Path(folder)/"two",synthetic=True)
            self.assertEqual([p.read_bytes() for p in first],[p.read_bytes() for p in second])

    def test_no_overwrite_or_partial_new_exports(self):
        fixture=evidence()
        with tempfile.TemporaryDirectory() as folder:
            existing=Path(folder)/"figure_1_architecture.pdf";existing.write_bytes(b"immutable")
            with self.assertRaises(FileExistsError):export_paper_figures(fixture,build_paper_tables(fixture),folder,synthetic=True)
            self.assertEqual(existing.read_bytes(),b"immutable")
            self.assertEqual(list(Path(folder).iterdir()),[existing])

    def test_observed_fixture_profiles_are_watermarked_and_missing_bias_is_not_plotted(self):
        fixture=evidence(contains_model_results=True,inference_tables={"profiles":[profile()]})
        with tempfile.TemporaryDirectory() as folder:
            paths=export_paper_figures(fixture,build_paper_tables(fixture),folder,synthetic=True)
            observed=[p for p in paths if p.parent.name=="observed_profiles"]
            self.assertEqual(len(observed),15)
            self.assertFalse((Path(folder)/"figure_3_accuracy_robustness_profiles.md").exists())
            self.assertFalse((Path(folder)/"figure_5_rdu_distribution.md").exists())
            self.assertTrue((Path(folder)/"figure_6_relation_vocabulary_bias.md").exists())
            self.assertIn("SYNTHETIC TEST DATA",next(p for p in observed if p.suffix==".svg").read_text())

    def test_bias_native_fixture_has_all_three_formats_and_interval(self):
        fixture=evidence(contains_model_results=True,lexical_tables={"answer_bias":[{
            "model_id":"fixture","query_relation":"before","q":.75,"n_valid":8,
            "q_wilson_low":.4,"q_wilson_high":.95,"gold_yes_planned":4,"gold_no_planned":4}]})
        with tempfile.TemporaryDirectory() as folder:
            paths=export_paper_figures(fixture,build_paper_tables(fixture),folder,synthetic=True)
            observed=[p for p in paths if "figure_6_relation_vocabulary_bias_" in p.name]
            self.assertEqual({p.suffix for p in observed},{".pdf",".svg",".png"})
            self.assertIn("N valid=8",next(p for p in observed if p.suffix==".svg").read_text())
            self.assertFalse((Path(folder)/"figure_6_relation_vocabulary_bias.md").exists())


if __name__ == "__main__":unittest.main()
