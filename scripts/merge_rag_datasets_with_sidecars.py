
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

STANDARD = (
    "documents.jsonl",
    "macros.jsonl",
    "units.jsonl",
    "relations.jsonl",
    "chunks.jsonl",
)

OPTIONAL = (
    "sections.jsonl",
    "formulas.jsonl",
    "formula_relations.jsonl",
    "quality_flags.jsonl",
    "unresolved_refs.jsonl",
)


def read_jsonl(path: Path, required: bool = True) -> list[dict]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise RuntimeError(f"{path}:{n}: objet JSON attendu")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    tmp.replace(path)


def ensure_unique(rows: list[dict], key: str, label: str) -> None:
    seen = set()
    for i, row in enumerate(rows, 1):
        value = row.get(key)
        if not value:
            raise RuntimeError(f"{label}[{i}]: {key} manquant")
        if value in seen:
            raise RuntimeError(f"{label}: doublon {key}={value}")
        seen.add(value)


def canon(obj: dict) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def normalized_relations(rows: list[dict]) -> tuple[list[dict], dict]:
    by_payload = {}
    old_ids = {}

    for row in rows:
        old_id = row.get("relation_id")
        payload = {
            k: v for k, v in row.items()
            if k != "relation_id"
        }
        pjson = canon(payload)
        if old_id:
            old_ids.setdefault(old_id, set()).add(pjson)
        if pjson not in by_payload:
            rid = "rel-" + hashlib.sha256(
                pjson.encode("utf-8")
            ).hexdigest()[:32]
            by_payload[pjson] = {
                "relation_id": rid,
                **payload,
            }

    out = list(by_payload.values())
    ensure_unique(out, "relation_id", "relations")

    collisions = {
        rid: len(payloads)
        for rid, payloads in old_ids.items()
        if len(payloads) > 1
    }
    return out, {
        "input": len(rows),
        "output": len(out),
        "exact_duplicates_removed": len(rows) - len(out),
        "old_id_collisions": len(collisions),
    }


def dedup_payload(rows: list[dict]) -> list[dict]:
    seen = {}
    for row in rows:
        seen.setdefault(canon(row), row)
    return list(seen.values())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fusionne plusieurs datasets RAG versionnés en préservant "
            "les sidecars mathématiques quand ils existent."
        )
    )
    ap.add_argument("--input", action="append", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(
            f"[ERREUR] Répertoire de sortie non vide: {args.out}"
        )

    merged = {name: [] for name in STANDARD + OPTIONAL}

    for src in args.input:
        print(f"[INPUT] {src}")
        for name in STANDARD:
            merged[name].extend(read_jsonl(src / name, required=True))
        for name in OPTIONAL:
            merged[name].extend(read_jsonl(src / name, required=False))

    ensure_unique(merged["documents.jsonl"], "document_id", "documents")
    ensure_unique(merged["units.jsonl"], "unit_id", "units")
    ensure_unique(merged["chunks.jsonl"], "chunk_id", "chunks")

    merged["relations.jsonl"], rel_stats = normalized_relations(
        merged["relations.jsonl"]
    )

    if merged["sections.jsonl"]:
        ensure_unique(
            merged["sections.jsonl"],
            "section_id",
            "sections",
        )
    if merged["formulas.jsonl"]:
        ensure_unique(
            merged["formulas.jsonl"],
            "formula_id",
            "formulas",
        )
    if merged["quality_flags.jsonl"]:
        ensure_unique(
            merged["quality_flags.jsonl"],
            "quality_flag_id",
            "quality_flags",
        )

    merged["formula_relations.jsonl"] = dedup_payload(
        merged["formula_relations.jsonl"]
    )
    merged["unresolved_refs.jsonl"] = dedup_payload(
        merged["unresolved_refs.jsonl"]
    )

    doc_ids = {r["document_id"] for r in merged["documents.jsonl"]}
    unit_ids = {r["unit_id"] for r in merged["units.jsonl"]}

    for u in merged["units.jsonl"]:
        if u.get("document_id") not in doc_ids:
            raise RuntimeError(f"unité sans document: {u['unit_id']}")

    for c in merged["chunks.jsonl"]:
        if c.get("unit_id") not in unit_ids:
            raise RuntimeError(f"chunk sans unité: {c['chunk_id']}")

    for r in merged["relations.jsonl"]:
        a = r.get("from_unit_id")
        b = r.get("to_unit_id")
        if a and a not in unit_ids:
            raise RuntimeError(f"relation from inconnue: {a}")
        if b and b not in unit_ids:
            raise RuntimeError(f"relation to inconnue: {b}")

    args.out.mkdir(parents=True, exist_ok=True)

    for name in STANDARD:
        write_jsonl(args.out / name, merged[name])
    for name in OPTIONAL:
        if merged[name]:
            write_jsonl(args.out / name, merged[name])

    stats = {
        "inputs": [str(x) for x in args.input],
        "documents": len(merged["documents.jsonl"]),
        "units": len(merged["units.jsonl"]),
        "chunks": len(merged["chunks.jsonl"]),
        "relations": len(merged["relations.jsonl"]),
        "chunk_types": dict(
            Counter(
                c.get("unit_type")
                for c in merged["chunks.jsonl"]
            )
        ),
        "content_kinds": dict(
            Counter(
                c.get("content_kind", "course")
                for c in merged["chunks.jsonl"]
            )
        ),
        "sidecars": {
            name: len(merged[name])
            for name in OPTIONAL
            if merged[name]
        },
        "relation_normalization": rel_stats,
    }

    (args.out / "merge_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[OK] Fusion créée: {args.out}")


if __name__ == "__main__":
    main()
