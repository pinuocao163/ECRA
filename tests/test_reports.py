import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_resources.py"
SPEC = importlib.util.spec_from_file_location("resource_report_audit", MODULE_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


BODY = """
           precision    recall  f1-score   support
           0     1.0000    0.5000    0.6667       2.0
           1     0.6667    1.0000    0.8000       2.0
    accuracy                         0.7500       4.0
   macro avg     0.8333    0.7500    0.7333       4.0
weighted avg     0.8333    0.7500    0.7333       4.0
"""


class ReportAuditTests(unittest.TestCase):
    def test_body_metrics_override_stale_headings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.txt"
            path.write_text(
                "seed1:===== TASK_99.00_99.00 =====\n"
                + BODY
                + "seed2:===== TASK_75.00_73.33 =====\n"
                + BODY
            )
            result = AUDIT.audit_reports(path)
        self.assertEqual(result["header_mismatch_seeds"], [1])
        self.assertAlmostEqual(result["summary"]["accuracy_percent"]["mean"], 75)
        self.assertAlmostEqual(result["summary"]["weighted_f1_percent"]["mean"], 73.33)
        self.assertAlmostEqual(result["summary"]["macro_f1_percent"]["mean"], 73.33)
        self.assertTrue(result["all_class_summaries_consistent"])
        self.assertEqual(result["identical_report_bodies"], [[1, 2]])

    def test_inconsistent_summary_is_detected(self):
        changed = BODY.replace("0.7333", "0.9000")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.txt"
            path.write_text(
                "seed1:===== TASK_75.00_90.00 =====\n"
                + changed
                + "seed2:===== TASK_75.00_73.33 =====\n"
                + BODY
            )
            result = AUDIT.audit_reports(path)
        self.assertFalse(result["all_class_summaries_consistent"])
        self.assertFalse(result["reports"][0]["class_summary_consistent_with_rounding"])

    def test_report_names_are_case_insensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "IEMOCAP.txt"
            path.write_text("")
            self.assertEqual(AUDIT.find_report(root, "iemocap"), path)
            (root / "iemocap.txt").write_text("")
            with self.assertRaises(ValueError):
                AUDIT.find_report(root, "iemocap")


if __name__ == "__main__":
    unittest.main()
