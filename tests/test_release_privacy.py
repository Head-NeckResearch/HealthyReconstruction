"""Check that the release gate rejects actual classes of accidental disclosure."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit_release.py"
_SPEC = importlib.util.spec_from_file_location("audit_release", _PATH)
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)


class ReleasePrivacyTests(unittest.TestCase):
    def test_only_source_and_documentation_paths_are_allowlisted(self):
        for name in ["data/recording.csv", "weights/model.pt", "results.json", "private.xlsx", "docs/face.png"]:
            self.assertFalse(_AUDIT.allowed_path(name))
        for name in ["README.md", "src/healthy_reconstruction/models.py", "config/pipeline.example.json"]:
            self.assertTrue(_AUDIT.allowed_path(name))

    def test_private_identifiers_are_checked_without_echoing_them(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root = base / "release"
            root.mkdir()
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            token = "private" + "SyntheticToken"
            (base / "identifiers.txt").write_text(token + "\n")
            (root / "README.md").write_text("# Example\n" + token + "\n")
            subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
            report = _AUDIT.audit(root, staged=True, private_identifiers=base / "identifiers.txt")
            self.assertFalse(report["passed"])
            self.assertEqual(report["findings"][0]["rule"], "private_identifier")
            self.assertNotIn(token, str(report))

    def test_staged_audit_catches_a_leak_removed_only_from_working_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            leaked_path = "/" + "home" + "/" + "private_person" + "/recordings"
            path = root / "README.md"
            path.write_text(leaked_path + "\n")
            subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
            path.write_text("# Clean working copy\n")
            self.assertTrue(_AUDIT.audit(root)["passed"])
            self.assertFalse(_AUDIT.audit(root, staged=True)["passed"])

    def test_private_review_applies_only_to_the_exact_reviewed_line(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root = base / "release"
            root.mkdir()
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            text = "GenericSymbol is a numerical variable."
            ids = base / "identifiers.txt"
            ids.write_text("GenericSymbol\n")
            review = base / "review.json"
            review.write_text(json.dumps({"reviewed_findings": [{
                "file": "README.md", "line": 1, "rule": "private_identifier",
                "classification": "semantic_false_positive", "reason": "Synthetic test of semantic review.",
                "line_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }]}))
            (root / "README.md").write_text(text + "\n")
            subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
            self.assertTrue(_AUDIT.audit(root, True, ids, review)["passed"])
            (root / "README.md").write_text(text + " Changed.\n")
            subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
            self.assertFalse(_AUDIT.audit(root, True, ids, review)["passed"])


if __name__ == "__main__":
    unittest.main()
