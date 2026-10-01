from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"JSON invalide {path}:{line_no}: {exc}"
                ) from exc
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True)
    args = ap.parse_args()

    docs = read_jsonl(args.processed / "documents.jsonl")
    units = read_jsonl(args.processed / "units.jsonl")
    rels = read_jsonl(args.processed / "relations.jsonl")
    chunks = read_jsonl(args.processed / "chunks.jsonl")
    macros = read_jsonl(args.processed / "macros.jsonl")
    envs = read_jsonl(
        args.processed / "declared_environments.jsonl"
    )
    stats = json.loads(
        (args.processed / "stats.json").read_text(
            encoding="utf-8"
        )
    )

    errors = []
    warnings = []

    doc_ids = [d.get("document_id") for d in docs]
    unit_ids = [u.get("unit_id") for u in units]
    chunk_ids = [c.get("chunk_id") for c in chunks]
    qdrant_ids = [c.get("qdrant_id") for c in chunks]

    for label, values in [
        ("document_id", doc_ids),
        ("unit_id", unit_ids),
        ("chunk_id", chunk_ids),
        ("qdrant_id", qdrant_ids),
    ]:
        if len(values) != len(set(values)):
            errors.append(f"Doublons {label}")

    doc_set = set(doc_ids)
    unit_set = set(unit_ids)

    for u in units:
        if u.get("document_id") not in doc_set:
            errors.append(
                f"Unité sans document: {u.get('unit_id')}"
            )
        if not u.get("exclude_from_answer_generation"):
            errors.append(
                f"Unité non exclue de la génération: "
                f"{u.get('unit_id')}"
            )

    for c in chunks:
        if c.get("unit_id") not in unit_set:
            errors.append(
                f"Chunk sans unité: {c.get('chunk_id')}"
            )
        if c.get("document_id") not in doc_set:
            errors.append(
                f"Chunk sans document: {c.get('chunk_id')}"
            )
        if c.get("content_kind") != "outline":
            errors.append(
                f"Chunk non-outline: {c.get('chunk_id')}"
            )
        if c.get("retrieval_role") != "topic_router":
            errors.append(
                f"Chunk non-router: {c.get('chunk_id')}"
            )
        if not c.get("exclude_from_answer_generation"):
            errors.append(
                f"Chunk autorisé pour génération: "
                f"{c.get('chunk_id')}"
            )
        if not c.get("embedding_text"):
            errors.append(
                f"embedding_text vide: {c.get('chunk_id')}"
            )

    allowed_relations = {"parent_of", "next_sibling"}
    for r in rels:
        if r.get("type") not in allowed_relations:
            errors.append(
                f"Relation non autorisée: "
                f"{r.get('relation_id')} {r.get('type')}"
            )
        if r.get("from_unit_id") not in unit_set:
            errors.append(
                f"Relation source inconnue: "
                f"{r.get('relation_id')}"
            )
        if r.get("to_unit_id") not in unit_set:
            errors.append(
                f"Relation cible inconnue: "
                f"{r.get('relation_id')}"
            )

    if stats.get("knowledge_chunks") != 0:
        errors.append("knowledge_chunks doit être 0")

    formal_usage = stats.get(
        "formal_environment_usage", {}
    )
    nonzero_formal = {
        k: v
        for k, v in formal_usage.items()
        if v
    }
    if nonzero_formal:
        warnings.append(
            f"Environnements formels utilisés: {nonzero_formal}"
        )

    if stats.get("content_display_math_blocks", 0):
        warnings.append(
            "Blocs mathématiques display présents dans la source"
        )

    if stats.get("content_refs", 0):
        warnings.append(
            "Références LaTeX présentes dans la source"
        )

    expected = {
        "documents": len(docs),
        "units": len(units),
        "outline_chunks": len(chunks),
        "relations": len(rels),
        "macro_declarations": len(macros),
        "declared_environments": len(envs),
    }
    for key, actual in expected.items():
        if stats.get(key) != actual:
            errors.append(
                f"stats[{key}]={stats.get(key)} != {actual}"
            )

    print("=== Smart Outline RAG Validation ===")
    print(
        f"documents: {len(docs)} "
        f"units: {len(units)} "
        f"chunks: {len(chunks)} "
        f"relations: {len(rels)}"
    )
    print(
        f"knowledge_chunks: {stats.get('knowledge_chunks')} "
        f"macros: {len(macros)} "
        f"declared_environments: {len(envs)}"
    )
    print(
        f"display_math: "
        f"{stats.get('content_display_math_blocks')} "
        f"refs: {stats.get('content_refs')}"
    )
    print(
        f"errors: {len(errors)} "
        f"warnings: {len(warnings)}"
    )

    for err in errors[:20]:
        print(f"[ERROR] {err}")
    for warning in warnings[:20]:
        print(f"[WARN] {warning}")

    if errors:
        raise SystemExit(1)

    print(
        "[OK] Navigation-only Smart skeleton dataset "
        "structurally valid"
    )
    print(
        "[DO NOT MERGE] Do not use these chunks as "
        "factual mathematical evidence."
    )


if __name__ == "__main__":
    main()
