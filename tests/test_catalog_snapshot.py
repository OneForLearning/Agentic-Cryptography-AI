import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

from livre_crypto_rag.catalog import load_catalog


SPEC = importlib.util.spec_from_file_location(
    "snapshot_6a2", Path(__file__).resolve().parents[1] / "scripts" / "snapshot_6a2.py"
)
SNAPSHOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SNAPSHOT)


class CatalogAndSnapshotTests(unittest.TestCase):
    def test_current_legacy_catalog_loads_and_resolves_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "corpus/catalog/sources.yaml"
            path.parent.mkdir(parents=True)
            source = {
                "id": "crypto-paillier-001", "title": "Paillier original", "author": None,
                "year": None, "lang": "en", "domain": "cryptographie", "level": "L3",
                "type": "article", "topics": ["paillier"],
                "file": "corpus/raw/crypto/paillier_original.pdf", "source_url": None,
            }
            path.write_text(yaml.safe_dump({"sources": [source]}), encoding="utf-8")
            pdf = root / source["file"]
            pdf.parent.mkdir(parents=True)
            pdf.write_bytes(b"%PDF-test")
            catalog = load_catalog(path, require_files=True, root=root)
            self.assertEqual(catalog.sources[0].pdf_path(root), pdf.resolve())
            self.assertEqual(catalog.sources[0].admission, "catalogue")
            self.assertEqual(catalog.sources[0].rights_status, "non_verifie")
            source["file"] = "corpus/raw/../../secret.pdf"
            path.write_text(yaml.safe_dump({"sources": [source]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_catalog(path, root=root)

    def test_snapshot_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_name = "corpus/raw/crypto/paillier.pdf"
            catalog = root / "corpus/catalog/sources.yaml"
            catalog.parent.mkdir(parents=True)
            catalog.write_text(yaml.safe_dump({"sources": [{"file": pdf_name}]}))
            for name in (*SNAPSHOT.FILES[1:], pdf_name):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((name + "\n").encode())
            output = root / "snapshots/6A2-test.zip"
            SNAPSHOT.snapshot(root, output)
            SNAPSHOT.verify(output, root)
            with self.assertRaises(FileExistsError):
                SNAPSHOT.snapshot(root, output)
            (root / SNAPSHOT.FILES[1]).write_text("modifié")
            SNAPSHOT.verify(output)
            with self.assertRaises(ValueError):
                SNAPSHOT.verify(output, root)
            bad = root / "snapshots/corrupt.zip"
            with zipfile.ZipFile(output) as original, zipfile.ZipFile(bad, "w") as modified:
                for name in original.namelist():
                    data = original.read(name)
                    modified.writestr(name, b"corrompu" if name == pdf_name else data)
            with self.assertRaises(ValueError):
                SNAPSHOT.verify(bad)


if __name__ == "__main__":
    unittest.main()
