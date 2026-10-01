"""Sauvegarde immuable et vérification des entrées de 6A.2.

Usage: uv run python scripts/snapshot_6a2.py create
       uv run python scripts/snapshot_6a2.py verify snapshots/6A2-....zip
       uv run python scripts/snapshot_6a2.py verify snapshots/6A2-....zip --compare-live
"""

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "corpus/catalog/sources.yaml",
    "corpus/processed/chunks.jsonl",
    "corpus/processed/chunks_test.jsonl",
    "corpus/processed/chunk_quality.jsonl",
    "datasets/cryptoexpert/candidates/candidates.jsonl",
    "datasets/cryptoexpert/reviewed/reviews.jsonl",
    "datasets/cryptoexpert/reviewed/review_history.jsonl",
    "datasets/cryptoexpert/human_review/decisions.jsonl",
    "datasets/cryptoexpert/approved/approved.jsonl",
    "datasets/cryptoexpert/system_prompt.txt",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot(root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Sauvegarde déjà présente : {output}")
    catalog = root / FILES[0]
    if not catalog.is_file():
        raise FileNotFoundError(catalog)
    try:
        import yaml
        source_files = [item["file"] for item in yaml.safe_load(catalog.read_text(encoding="utf-8"))["sources"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Catalogue invalide : PDF impossibles à inventorier") from exc
    selected = sorted(set(FILES) | set(source_files))
    data = {}
    root_real = root.resolve()
    for name in selected:
        path = (root / name).resolve()
        if not path.is_relative_to(root_real):
            raise ValueError(f"Chemin hors projet : {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Fichier 6A.2 absent : {name}")
        data[name] = path.read_bytes()
    manifest = {
        "profile": "Livre_Crypto_6A.2_snapshot_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: {"sha256": digest(content), "size": len(content)}
                  for name, content in data.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".6a2-", suffix=".zip", dir=output.parent, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in data.items():
                archive.writestr(name, content)
            archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True).encode())
        verify(temp)
        # Publication exclusive : jamais d'écrasement d'une sauvegarde existante.
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Sauvegarde vérifiée : {output} ({len(data)} fichiers)")


def verify(archive_path: Path, root: Path | None = None) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Entrées ZIP dupliquées")
        manifest = json.loads(archive.read("MANIFEST.json"))
        expected = set(manifest["files"]) | {"MANIFEST.json"}
        if set(names) != expected:
            raise ValueError("Liste des fichiers différente du manifeste")
        for name, record in manifest["files"].items():
            data = archive.read(name)
            if digest(data) != record["sha256"] or len(data) != record["size"]:
                raise ValueError(f"Empreinte ou taille invalide : {name}")
            if root is not None and (not (root / name).is_file()
                                     or digest((root / name).read_bytes()) != record["sha256"]):
                raise ValueError(f"Fichier courant différent : {name}")
    print(f"Intégrité OK : {archive_path} ({len(manifest['files'])} fichiers)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("archive", nargs="?", type=Path)
    parser.add_argument("--compare-live", action="store_true")
    args = parser.parse_args()
    if args.action == "create":
        if args.archive is not None or args.compare_live:
            parser.error("create n'accepte ni archive ni --compare-live")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot(ROOT, ROOT / "snapshots" / f"6A2-{stamp}.zip")
    else:
        if args.archive is None:
            parser.error("verify exige un chemin d'archive")
        verify(args.archive, ROOT if args.compare_live else None)


if __name__ == "__main__":
    main()
