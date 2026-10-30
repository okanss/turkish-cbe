"""Structural shortcut and read-only audit publication contracts."""

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from research_audit.data_checks import endpoint_shortcut
from research_audit.__main__ import audit,CHECKS
from data_generation.generator import GenerationPlan,generate_record


class AuditConstructTests(unittest.TestCase):
    def test_degree_shortcut_solves_endpoints_at_every_registered_size(self):
        for n in (4,8,16,32,64,128,256,512,1024):
            plan=GenerationPlan(seed=771,sizes=(n,),examples_per_size=4,entity_types=('symbolic_ids',))
            for i in range(4):
                row=generate_record(plan,n,i,'endpoint')
                self.assertEqual(endpoint_shortcut(list(reversed(row['premise_edges'])),row['query_subject'],row['query_object'],row['query_relation']),row['gold_label'])

    def test_shortcut_does_not_claim_to_solve_internal_queries(self):
        edges=[('D','B'),('B','A'),('A','C')]
        self.assertIsNone(endpoint_shortcut(edges,'B','A','before'))
        self.assertTrue(endpoint_shortcut(edges,'D','C','before'))
        self.assertFalse(endpoint_shortcut(edges,'C','D','before'))
        self.assertTrue(endpoint_shortcut(edges,'C','D','after'))

    def test_audit_has_all_18_requested_checks_with_scoped_human_limits(self):
        self.assertEqual([r[0] for r in CHECKS],list(range(1,19)))
        self.assertTrue(any('human_pending' in r[2] for r in CHECKS))
        self.assertTrue(any('no_real' in r[2] for r in CHECKS))

    def test_audit_cannot_overwrite_or_escape_output_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();(root/'reports/audit/exists').mkdir(parents=True)
            for path in ('reports/audit/exists','src/new','../escape'):
                with self.subTest(path=path),patch('research_audit.__main__.run_data_checks') as data:
                    with self.assertRaises(ValueError):audit(root,path)
                    data.assert_not_called()

    def test_changed_installed_audit_source_is_rejected(self):
        project=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            shutil.copytree(project/'reports/research_audit',root/'reports/research_audit',ignore=shutil.ignore_patterns('__pycache__'))
            (root/'reports/research_audit/__init__.py').write_text('changed=True\n')
            with self.assertRaisesRegex(ValueError,'installed audit sources'):audit(root,'reports/audit/new')


if __name__=='__main__':unittest.main()
