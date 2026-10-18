from pathlib import Path
import tempfile
import unittest

from scripts.audit_japanese_resources import audit_file, assert_physical, copy_japanese_bundles


class JapaneseResourceAuditTests(unittest.TestCase):
    def test_source_copy_hashes_match(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, target = base / "source", base / "target"
            relative = "models/romajiASR/phoneme_vocab.json"
            for root in (source, target):
                (root / relative).parent.mkdir(parents=True)
                (root / relative).write_text('{"tokens": ["a", "ka"]}', encoding="utf-8")
            result = audit_file(relative, target, source)
            self.assertTrue(result["passed"])
            self.assertEqual(result["link_count"], 1)
            (target / relative).write_text('{"tokens": ["a"]}', encoding="utf-8")
            self.assertFalse(audit_file(relative, target, source)["passed"])

    def test_missing_resource_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(audit_file("missing.onnx", root, root)["passed"])

    def test_hard_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.onnx"
            original.write_bytes(b"test model")
            linked = root / "linked.onnx"
            linked.hardlink_to(original)
            with self.assertRaisesRegex(ValueError, "Hard-linked"):
                assert_physical(linked, root)

    def test_resource_outside_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            outside = Path(directory) / "outside.onnx"
            outside.write_bytes(b"test model")
            with self.assertRaisesRegex(ValueError, "escapes"):
                assert_physical(outside, root)

    def test_missing_source_bundle_fails_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                copy_japanese_bundles(root / "target", root / "missing_source")


if __name__ == "__main__":
    unittest.main()
