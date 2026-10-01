from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

FILES = (
    "documents.jsonl",
    "macros.jsonl",
    "units.jsonl",
    "relations.jsonl",
    "chunks.jsonl",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    out: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"JSON invalide {path}:{n}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise RuntimeError(
                    f"Objet JSON attendu {path}:{n}"
                )

            out.append(obj)

    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    tmp.replace(path)


def ensure_unique(
    records: list[dict],
    key: str,
    label: str,
) -> None:
    seen: dict[str, int] = {}

    for pos, record in enumerate(records, 1):
        value = record.get(key)

        if not value:
            raise RuntimeError(
                f"{label}: {key} manquant à l'entrée {pos}"
            )

        if value in seen:
            raise RuntimeError(
                f"{label}: doublon {key}={value} "
                f"(entrées {seen[value]} et {pos})"
            )

        seen[value] = pos


def canonical_relation_payload(
    relation: dict,
) -> dict:
    """
    Relation identity independent from relation_id.

    The old datasets can legitimately contain relation_id collisions because
    different ingestion pipelines generated 20-hex SHA prefixes in their own
    namespace.  relation_id is not used as a foreign key by the RAG pipeline;
    the meaningful relation is its payload (type/from/to/label/document/etc.).
    """
    return {
        key: value
        for key, value in relation.items()
        if key != "relation_id"
    }


def canonical_json(obj: dict) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def new_relation_id(payload: dict) -> str:
    digest = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()

    # 32 hex chars = 128 bits; much larger than the previous 80-bit prefix.
    return "rel-" + digest[:32]


def normalize_relations(
    relation_records: list[tuple[str, dict]],
) -> tuple[list[dict], dict]:
    """
    - Deduplicates exact duplicate relation payloads.
    - Regenerates every relation_id deterministically from the full payload.
    - Reports old-ID collisions instead of silently hiding them.
    """
    old_id_payloads: dict[str, set[str]] = {}
    payload_to_relation: dict[str, dict] = {}
    exact_duplicates = 0

    for source_name, relation in relation_records:
        old_id = relation.get("relation_id")

        if not old_id:
            raise RuntimeError(
                f"relations: relation_id manquant dans {source_name}"
            )

        payload = canonical_relation_payload(relation)
        payload_json = canonical_json(payload)

        old_id_payloads.setdefault(
            old_id,
            set(),
        ).add(payload_json)

        if payload_json in payload_to_relation:
            exact_duplicates += 1
            continue

        normalized = dict(payload)
        normalized["relation_id"] = new_relation_id(payload)
        payload_to_relation[payload_json] = normalized

    relations = list(payload_to_relation.values())

    # Collision under the new 128-bit ID would indicate either an impossible
    # hash collision or a bug in canonicalization. Fail closed.
    ensure_unique(
        relations,
        "relation_id",
        "relations normalisées",
    )

    old_id_collisions = {
        old_id: len(payloads)
        for old_id, payloads in old_id_payloads.items()
        if len(payloads) > 1
    }

    stats = {
        "input_relation_records": len(relation_records),
        "output_relations": len(relations),
        "exact_duplicate_relations_removed": exact_duplicates,
        "old_relation_id_collisions": len(old_id_collisions),
        "old_relation_id_collision_examples": dict(
            list(old_id_collisions.items())[:20]
        ),
    }

    return relations, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Merge immutable processed RAG datasets into a new version. "
            "Safely normalizes relation IDs and deduplicates exact relations."
        )
    )

    ap.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Processed dataset directory. Repeatable.",
    )

    ap.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(
            f"[ERREUR] Sortie non vide : {args.out}. "
            "Supprime-la si elle provient d'une fusion échouée, "
            "ou utilise un nouveau répertoire versionné."
        )

    merged = {
        name: []
        for name in FILES
        if name != "relations.jsonl"
    }

    relation_records: list[tuple[str, dict]] = []

    for src in args.input:
        print(f"[INPUT] {src}")

        for name in FILES:
            records = read_jsonl(src / name)

            if name == "relations.jsonl":
                relation_records.extend(
                    (str(src), record)
                    for record in records
                )
            else:
                merged[name].extend(records)

    # IDs referenced elsewhere must remain strictly unique.
    ensure_unique(
        merged["documents.jsonl"],
        "document_id",
        "documents",
    )
    ensure_unique(
        merged["units.jsonl"],
        "unit_id",
        "units",
    )
    ensure_unique(
        merged["chunks.jsonl"],
        "chunk_id",
        "chunks",
    )

    relations, relation_stats = normalize_relations(
        relation_records
    )
    merged["relations.jsonl"] = relations

    doc_ids = {
        doc["document_id"]
        for doc in merged["documents.jsonl"]
    }
    unit_ids = {
        unit["unit_id"]
        for unit in merged["units.jsonl"]
    }

    for unit in merged["units.jsonl"]:
        if unit.get("document_id") not in doc_ids:
            raise RuntimeError(
                f"Unité sans document : {unit['unit_id']}"
            )

    for chunk in merged["chunks.jsonl"]:
        if chunk.get("unit_id") not in unit_ids:
            raise RuntimeError(
                f"Chunk sans unité : {chunk['chunk_id']}"
            )

    dangling_relations = []

    for relation in merged["relations.jsonl"]:
        from_unit = relation.get("from_unit_id")
        to_unit = relation.get("to_unit_id")

        if from_unit and from_unit not in unit_ids:
            dangling_relations.append(
                (
                    relation["relation_id"],
                    "from_unit_id",
                    from_unit,
                )
            )

        if (
            to_unit
            and relation.get("resolved", True)
            and to_unit not in unit_ids
        ):
            dangling_relations.append(
                (
                    relation["relation_id"],
                    "to_unit_id",
                    to_unit,
                )
            )

    if dangling_relations:
        sample = dangling_relations[:10]
        raise RuntimeError(
            "Relations pendantes détectées. "
            f"Exemples: {sample}"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in FILES:
        write_jsonl(
            args.out / name,
            merged[name],
        )

    stats = {
        "inputs": [
            str(path)
            for path in args.input
        ],
        "documents": len(
            merged["documents.jsonl"]
        ),
        "macros": len(
            merged["macros.jsonl"]
        ),
        "units": len(
            merged["units.jsonl"]
        ),
        "relations": len(
            merged["relations.jsonl"]
        ),
        "chunks": len(
            merged["chunks.jsonl"]
        ),
        "chunk_types": dict(
            Counter(
                chunk.get("unit_type")
                for chunk in merged["chunks.jsonl"]
            )
        ),
        "content_kinds": dict(
            Counter(
                chunk.get(
                    "content_kind",
                    "course",
                )
                for chunk in merged["chunks.jsonl"]
            )
        ),
        "relation_normalization": relation_stats,
    }

    (
        args.out / "merge_stats.json"
    ).write_text(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        f"[OK] Dataset fusionné : "
        f"{args.out.resolve()}"
    )


if __name__ == "__main__":
    main()
