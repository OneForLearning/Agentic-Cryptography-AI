
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise RuntimeError(f"{path}:{n}: object JSON attendu")
            rows.append(obj)
    return rows


def unique(rows: list[dict], key: str, errors: list[str], label: str) -> set[str]:
    seen = set()
    for i, row in enumerate(rows, 1):
        value = row.get(key)
        if not value:
            errors.append(f"{label}[{i}] {key} manquant")
            continue
        if value in seen:
            errors.append(f"{label}: doublon {key}={value}")
        seen.add(value)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True)
    args = ap.parse_args()

    p = args.processed
    documents = read_jsonl(p / "documents.jsonl")
    sections = read_jsonl(p / "sections.jsonl")
    units = read_jsonl(p / "units.jsonl")
    formulas = read_jsonl(p / "formulas.jsonl")
    relations = read_jsonl(p / "relations.jsonl")
    chunks = read_jsonl(p / "chunks.jsonl")
    qflags = read_jsonl(p / "quality_flags.jsonl")
    unresolved = read_jsonl(p / "unresolved_refs.jsonl")

    errors: list[str] = []
    warnings: list[str] = []

    doc_ids = unique(documents, "document_id", errors, "documents")
    section_ids = unique(sections, "section_id", errors, "sections")
    unit_ids = unique(units, "unit_id", errors, "units")
    formula_ids = unique(formulas, "formula_id", errors, "formulas")
    relation_ids = unique(relations, "relation_id", errors, "relations")
    chunk_ids = unique(chunks, "chunk_id", errors, "chunks")
    quality_ids = unique(qflags, "quality_flag_id", errors, "quality_flags")

    for u in units:
        if u.get("document_id") not in doc_ids:
            errors.append(f"unit sans document: {u.get('unit_id')}")
        sid = u.get("section_id")
        if sid and sid not in section_ids:
            errors.append(f"unit avec section inconnue: {u.get('unit_id')} -> {sid}")

    for f in formulas:
        if f.get("unit_id") not in unit_ids:
            errors.append(f"formule sans unité: {f.get('formula_id')}")

    for c in chunks:
        if c.get("unit_id") not in unit_ids:
            errors.append(f"chunk sans unité: {c.get('chunk_id')}")
        if c.get("document_id") not in doc_ids:
            errors.append(f"chunk sans document: {c.get('chunk_id')}")
        if len(c.get("latex", "")) > 6000:
            errors.append(
                f"chunk > 6000 caractères: {c.get('chunk_id')} "
                f"({len(c.get('latex',''))})"
            )
        for fid in c.get("formula_ids", []):
            if fid not in formula_ids:
                errors.append(f"chunk {c.get('chunk_id')} -> formule inconnue {fid}")
        for qid in c.get("quality_flags", []):
            if qid not in quality_ids:
                errors.append(f"chunk {c.get('chunk_id')} -> quality flag inconnu {qid}")

    proof_count = 0
    linked_proof_count = 0

    for u in units:
        if u.get("unit_type") == "proof":
            proof_count += 1
            if u.get("proof_of"):
                linked_proof_count += 1
                if u["proof_of"] not in unit_ids:
                    errors.append(
                        f"preuve {u['unit_id']} -> statement inconnu {u['proof_of']}"
                    )
            else:
                errors.append(f"preuve non reliée: {u['unit_id']}")

    for rel in relations:
        a = rel.get("from_unit_id")
        b = rel.get("to_unit_id")
        if a and a not in unit_ids:
            errors.append(f"relation {rel['relation_id']} from inconnu {a}")
        if b and b not in unit_ids:
            errors.append(f"relation {rel['relation_id']} to inconnu {b}")

    source_ids = [
        u["source_unit_id"]
        for u in units
        if u.get("source_unit_id")
    ]
    if len(source_ids) != len(set(source_ids)):
        errors.append("source_unit_id dupliqué")

    labels = []
    refs = []
    for u in units:
        labels.extend(u.get("labels", []))
        refs.extend(u.get("refs", []))

    if unresolved:
        errors.append(f"{len(unresolved)} références non résolues")

    severe_flags = [
        q for q in qflags
        if q.get("severity") == "critical_for_answering"
    ]
    brace_flags = [
        q for q in qflags
        if q.get("type") == "latex_brace_imbalance"
    ]
    if severe_flags:
        warnings.append(
            f"{len(severe_flags)} quality flags critiques pour la réponse; "
            "les chunks concernés doivent déclencher une revue humaine"
        )
    if brace_flags:
        warnings.append(
            f"{len(brace_flags)} unités avec déséquilibre lexical des accolades LaTeX"
        )

    print("=== Maths Structured RAG Validation ===")
    print(
        f"documents: {len(documents)} sections: {len(sections)} "
        f"units: {len(units)} chunks: {len(chunks)}"
    )
    print(
        f"formulas: {len(formulas)} relations: {len(relations)} "
        f"refs: {len(refs)} unresolved: {len(unresolved)}"
    )
    print(
        f"proofs: {proof_count} linked: {linked_proof_count}"
    )
    print(
        f"unit_types: {dict(Counter(u.get('unit_type') for u in units))}"
    )
    print(
        f"semantic_roles: {dict(Counter(u.get('semantic_role') for u in units))}"
    )
    print(
        f"quality_flags: {len(qflags)} "
        f"({dict(Counter(q.get('type') for q in qflags))})"
    )
    print(
        f"errors: {len(errors)} warnings: {len(warnings)}"
    )

    for w in warnings:
        print("[WARN]", w)
    for e in errors[:50]:
        print("[ERROR]", e)

    if errors:
        raise SystemExit(1)

    print("[OK] Maths structured dataset valid")


if __name__ == "__main__":
    main()
