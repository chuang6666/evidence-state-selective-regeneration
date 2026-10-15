import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import uuid
from contextlib import contextmanager

from evidence_state_repro.offline import ranking, safe_output


@contextmanager
def temporary_workspace():
    base = Path(os.environ.get("EVIDENCE_STATE_TEST_TMPDIR", tempfile.gettempdir())).resolve()
    folder = base / ("evidence-state-tests-" + uuid.uuid4().hex)
    folder.mkdir()  # inherit ACLs; Windows sandbox rejects mkdtemp's mode=0700
    try:
        yield folder
    finally:
        if folder.resolve().parent != base or not folder.name.startswith("evidence-state-tests-"):
            raise RuntimeError("Unsafe temporary cleanup path")
        shutil.rmtree(folder)


class OfflineExtensionTests(unittest.TestCase):
    def test_tied_score_AP_and_AUC(self):
        data = [{"qid": q, "initial_wrong_label": label, "scores": {"X": score}, "saved_selected": {"X": q == "a"}}
                for q, label, score in (("a", 1, .9), ("b", 0, .9), ("c", 1, .2), ("d", 0, .0))]
        result = ranking(data, "X")
        self.assertAlmostEqual(result["AP_percent"], (0.25 + 1 / 3) * 100)
        self.assertAlmostEqual(result["ROC_AUC_percent"], 62.5)
        self.assertEqual(result["errors"], 1)

    def test_output_inside_repo_is_rejected(self):
        with temporary_workspace() as folder:
            with self.assertRaises(ValueError):
                safe_output(Path(folder), Path(folder) / "references")

    def test_nonempty_output_is_preserved(self):
        with temporary_workspace() as folder:
            root, out = Path(folder) / "repo", Path(folder) / "out"
            root.mkdir()
            out.mkdir()
            (out / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                safe_output(root, out)
            self.assertEqual((out / "keep.txt").read_text(encoding="utf-8"), "keep")

    def test_statistical_bodies_match_provenance(self):
        root = Path(__file__).resolve().parents[1]
        for entry in json.loads((root / "provenance/statistics_functions.json").read_text(encoding="utf-8")):
            source = (root / entry["candidate_path"]).read_text(encoding="utf-8")
            functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
            for function in entry["functions"]:
                body = ast.get_source_segment(source, functions[function["name"]])
                self.assertEqual(hashlib.sha256(body.encode()).hexdigest(), function["body_sha256"])

    def test_public_predictions_have_no_reference_fields(self):
        root = Path(__file__).resolve().parents[1]
        for file in (root / "frozen_predictions").glob("*.jsonl"):
            with file.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    self.assertEqual(set(row), {"qid", "methods"})
                    self.assertIsInstance(row["methods"], dict)


if __name__ == "__main__":
    unittest.main()
