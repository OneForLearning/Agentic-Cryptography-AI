from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FILES = ("documents.jsonl", "macros.jsonl", "units.jsonl", "relations.jsonl", "chunks.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    out = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSON invalide {path}:{n}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"Objet JSON attendu {path}:{n}")
            out.append(obj)
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def ensure_unique(records: list[dict], key: str, label: str) -> None:
    seen = set()
    for r in records:
        value = r.get(key)
        if not value:
            raise RuntimeError(f"{label}: {key} manquant")
        if value in seen:
            raise RuntimeError(f"{label}: doublon {key}={value}")
        seen.add(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge immutable processed RAG datasets into a new version.")
    ap.add_argument("--input", action="append", type=Path, required=True, help="Processed dataset directory. Repeatable.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"[ERREUR] Sortie non vide : {args.out}. Utilise un nouveau répertoire versionné.")

    merged = {name: [] for name in FILES}
    for src in args.input:
        print(f"[INPUT] {src}")
        for name in FILES:
            merged[name].extend(read_jsonl(src / name))

    ensure_unique(merged["documents.jsonl"], "document_id", "documents")
    ensure_unique(merged["units.jsonl"], "unit_id", "units")
    ensure_unique(merged["relations.jsonl"], "relation_id", "relations")
    ensure_unique(merged["chunks.jsonl"], "chunk_id", "chunks")

    doc_ids = {d["document_id"] for d in merged["documents.jsonl"]}
    unit_ids = {u["unit_id"] for u in merged["units.jsonl"]}
    for u in merged["units.jsonl"]:
        if u.get("document_id") not in doc_ids:
            raise RuntimeError(f"Unité sans document : {u['unit_id']}")
    for c in merged["chunks.jsonl"]:
        if c.get("unit_id") not in unit_ids:
            raise RuntimeError(f"Chunk sans unité : {c['chunk_id']}")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in merged.items():
        write_jsonl(args.out / name, records)

    stats = {
        "inputs": [str(x) for x in args.input],
        "documents": len(merged["documents.jsonl"]),
        "macros": len(merged["macros.jsonl"]),
        "units": len(merged["units.jsonl"]),
        "relations": len(merged["relations.jsonl"]),
        "chunks": len(merged["chunks.jsonl"]),
        "chunk_types": dict(Counter(c.get("unit_type") for c in merged["chunks.jsonl"])),
        "content_kinds": dict(Counter(c.get("content_kind", "course") for c in merged["chunks.jsonl"])),
    }
    (args.out / "merge_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[OK] Dataset fusionné : {args.out.resolve()}")


if __name__ == "__main__":
    main()
