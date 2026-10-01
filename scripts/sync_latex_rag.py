from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"JSON invalide {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise RuntimeError(
                    f"Objet JSON attendu {path}:{line_no}"
                )
            out.append(obj)
    return out


def batched(seq, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def canonical_json(obj: dict) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_hash(payload: dict) -> str:
    return sha256_text(canonical_json(payload))


def response_embeddings(resp):
    if isinstance(resp, dict):
        value = resp.get("embeddings") or resp.get("embedding")
        if value is not None:
            if value and isinstance(value[0], (int, float)):
                return [value]
            return value

    if hasattr(resp, "embeddings"):
        return resp.embeddings

    if hasattr(resp, "embedding"):
        return [resp.embedding]

    raise RuntimeError("Réponse Ollama embeddings inconnue")


def vector_from_record(record):
    vector = getattr(record, "vector", None)

    if vector is None:
        raise RuntimeError(
            f"Vecteur absent pour le point Qdrant {getattr(record, 'id', '?')}"
        )

    if isinstance(vector, dict):
        if len(vector) != 1:
            raise RuntimeError(
                "Collection Qdrant multi-vecteurs non prise en charge automatiquement"
            )
        vector = next(iter(vector.values()))

    return vector


def validate_current_chunks(chunks: list[dict]) -> dict[str, dict]:
    by_id: dict[str, dict] = {}

    required = (
        "chunk_id",
        "unit_id",
        "document_id",
        "unit_type",
        "embedding_text",
        "latex",
        "qdrant_id",
    )

    for i, chunk in enumerate(chunks, 1):
        missing = [key for key in required if key not in chunk]
        if missing:
            raise RuntimeError(
                f"Chunk #{i}: champs manquants: {missing}"
            )

        cid = str(chunk["chunk_id"])
        if cid in by_id:
            raise RuntimeError(
                f"Doublon chunk_id dans le corpus courant : {cid}"
            )

        by_id[cid] = chunk

    return by_id


def scan_qdrant(qclient, collection: str) -> tuple[dict[str, dict], list[str]]:
    """
    Scan payloads only: no vectors are transferred.
    Qdrant itself is the source of truth for what is already indexed.
    """
    existing: dict[str, dict] = {}
    duplicates: list[str] = []
    offset = None

    while True:
        records, next_offset = qclient.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for rec in records:
            payload = getattr(rec, "payload", {}) or {}
            cid = payload.get("chunk_id")

            if not cid:
                continue

            cid = str(cid)

            info = {
                "point_id": getattr(rec, "id"),
                "payload": payload,
                "embedding_hash": (
                    sha256_text(payload["embedding_text"])
                    if isinstance(payload.get("embedding_text"), str)
                    else None
                ),
                "payload_hash": payload_hash(payload),
            }

            if cid in existing:
                duplicates.append(cid)
            else:
                existing[cid] = info

        if next_offset is None:
            break

        offset = next_offset

    return existing, duplicates


def classify_qdrant(
    current: dict[str, dict],
    existing: dict[str, dict],
) -> dict[str, list]:
    result = {
        "new": [],
        "embedding_changed": [],
        "payload_only": [],
        "unchanged": [],
        "id_migration": [],
        "stale": [],
    }

    for cid, chunk in current.items():
        old = existing.get(cid)

        if old is None:
            result["new"].append(chunk)
            continue

        current_embed_hash = sha256_text(chunk["embedding_text"])
        current_payload_hash = payload_hash(chunk)

        current_qid = str(chunk["qdrant_id"])
        old_qid = str(old["point_id"])

        if current_qid != old_qid:
            result["id_migration"].append((chunk, old))
            continue

        if old["embedding_hash"] != current_embed_hash:
            result["embedding_changed"].append((chunk, old))
        elif old["payload_hash"] != current_payload_hash:
            result["payload_only"].append((chunk, old))
        else:
            result["unchanged"].append(chunk)

    current_ids = set(current)
    for cid, old in existing.items():
        if cid not in current_ids:
            result["stale"].append((cid, old))

    return result


def create_qdrant_collection(qclient, models, collection: str, dim: int):
    qclient.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
        ),
    )

    for field in [
        "chunk_id",
        "document_id",
        "unit_type",
        "language",
        "source_file",
        "label",
        "content_hash",
    ]:
        try:
            qclient.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def point_payload(chunk: dict) -> dict:
    # Keep exact compatibility with index_latex_rag.py:
    # every chunk field is stored in the Qdrant payload.
    return dict(chunk)


def embed_and_upsert(
    *,
    qclient,
    models,
    oclient,
    collection: str,
    model_name: str,
    entries: list[tuple[dict, object | None]],
    batch_size: int,
):
    """
    entries: (chunk, old_info_or_none)
    old_info is only informational. Current qdrant_id is used.
    """
    done = 0
    total = len(entries)

    for batch in batched(entries, batch_size):
        chunks = [item[0] for item in batch]
        texts = [chunk["embedding_text"] for chunk in chunks]

        response = oclient.embed(
            model=model_name,
            input=texts,
        )
        vectors = response_embeddings(response)

        if len(vectors) != len(chunks):
            raise RuntimeError(
                "Nombre d'embeddings différent du nombre de chunks"
            )

        points = [
            models.PointStruct(
                id=chunk["qdrant_id"],
                vector=vector,
                payload=point_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        qclient.upsert(
            collection_name=collection,
            points=points,
            wait=True,
        )

        done += len(batch)
        print(f"Embedded+upsert {done}/{total}")


def reuse_vectors_and_upsert(
    *,
    qclient,
    models,
    collection: str,
    entries: list[tuple[dict, dict]],
    batch_size: int,
):
    """
    Payload changed but embedding did not, or point ID needs migration.
    Retrieve old vectors from Qdrant and re-use them: zero Ollama work.
    """
    done = 0
    total = len(entries)

    for batch in batched(entries, batch_size):
        old_ids = [item[1]["point_id"] for item in batch]

        records = qclient.retrieve(
            collection_name=collection,
            ids=old_ids,
            with_payload=False,
            with_vectors=True,
        )

        by_id = {
            str(getattr(record, "id")): record
            for record in records
        }

        points = []
        old_ids_to_delete = []

        for chunk, old in batch:
            old_id = old["point_id"]
            record = by_id.get(str(old_id))

            if record is None:
                raise RuntimeError(
                    f"Point Qdrant introuvable pendant la mise à jour : {old_id}"
                )

            vector = vector_from_record(record)

            points.append(
                models.PointStruct(
                    id=chunk["qdrant_id"],
                    vector=vector,
                    payload=point_payload(chunk),
                )
            )

            if str(old_id) != str(chunk["qdrant_id"]):
                old_ids_to_delete.append(old_id)

        qclient.upsert(
            collection_name=collection,
            points=points,
            wait=True,
        )

        if old_ids_to_delete:
            qclient.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(
                    points=old_ids_to_delete
                ),
                wait=True,
            )

        done += len(batch)
        print(f"Vector-reuse upsert {done}/{total}")


def ensure_sqlite_schema(con: sqlite3.Connection):
    con.execute("PRAGMA journal_mode=WAL")

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            unit_id TEXT,
            document_id TEXT,
            unit_type TEXT,
            payload_json TEXT NOT NULL,
            embedding_text TEXT NOT NULL
        )
        """
    )

    con.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            document_id UNINDEXED,
            unit_type UNINDEXED,
            embedding_text,
            tokenize='unicode61'
        )
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS relations (
            type TEXT,
            from_unit_id TEXT,
            to_unit_id TEXT,
            target_label TEXT,
            resolved INTEGER
        )
        """
    )

    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_unit ON chunks(unit_id)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_rel_from ON relations(from_unit_id)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_rel_to ON relations(to_unit_id)"
    )


def sync_sqlite(
    db_path: Path,
    chunks: list[dict],
    relations: list[dict],
    *,
    prune: bool,
) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    try:
        ensure_sqlite_schema(con)

        existing_rows = con.execute(
            "SELECT chunk_id, payload_json FROM chunks"
        ).fetchall()

        existing = {
            row["chunk_id"]: row["payload_json"]
            for row in existing_rows
        }

        current_ids = {str(chunk["chunk_id"]) for chunk in chunks}

        inserted = 0
        updated = 0
        unchanged = 0

        for chunk in chunks:
            cid = str(chunk["chunk_id"])
            payload_json = json.dumps(
                chunk,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            old_payload = existing.get(cid)

            if old_payload == payload_json:
                unchanged += 1
                continue

            if old_payload is None:
                inserted += 1
            else:
                updated += 1

            con.execute(
                """
                INSERT INTO chunks(
                    chunk_id, unit_id, document_id, unit_type,
                    payload_json, embedding_text
                )
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    unit_id=excluded.unit_id,
                    document_id=excluded.document_id,
                    unit_type=excluded.unit_type,
                    payload_json=excluded.payload_json,
                    embedding_text=excluded.embedding_text
                """,
                (
                    cid,
                    chunk["unit_id"],
                    chunk["document_id"],
                    chunk["unit_type"],
                    payload_json,
                    chunk["embedding_text"],
                ),
            )

            con.execute(
                "DELETE FROM chunks_fts WHERE chunk_id=?",
                (cid,),
            )

            con.execute(
                """
                INSERT INTO chunks_fts(
                    chunk_id, document_id, unit_type, embedding_text
                )
                VALUES(?,?,?,?)
                """,
                (
                    cid,
                    chunk["document_id"],
                    chunk["unit_type"],
                    chunk["embedding_text"],
                ),
            )

        stale = [
            cid
            for cid in existing
            if cid not in current_ids
        ]

        deleted = 0
        if prune and stale:
            for cid in stale:
                con.execute(
                    "DELETE FROM chunks WHERE chunk_id=?",
                    (cid,),
                )
                con.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id=?",
                    (cid,),
                )
                deleted += 1

        # Relations are cheap compared with embeddings.
        # Rebuild them from the authoritative processed dataset each sync.
        con.execute("DELETE FROM relations")

        con.executemany(
            """
            INSERT INTO relations(
                type, from_unit_id, to_unit_id, target_label, resolved
            )
            VALUES(?,?,?,?,?)
            """,
            [
                (
                    relation.get("type"),
                    relation.get("from_unit_id"),
                    relation.get("to_unit_id"),
                    relation.get("target_label"),
                    1 if relation.get("resolved", True) else 0,
                )
                for relation in relations
            ],
        )

        con.commit()

        return {
            "inserted": inserted,
            "updated": updated,
            "unchanged": unchanged,
            "stale": len(stale),
            "deleted": deleted,
            "relations": len(relations),
        }

    finally:
        con.close()


def write_sync_state(
    path: Path,
    *,
    args,
    current_chunks: dict[str, dict],
    qdrant_count,
    classification,
    sqlite_stats,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "processed": str(args.processed),
        "collection": args.collection,
        "embedding_model": args.model,
        "corpus_chunks": len(current_chunks),
        "qdrant_points": qdrant_count,
        "counts": {
            "new": len(classification["new"]),
            "embedding_changed": len(classification["embedding_changed"]),
            "payload_only": len(classification["payload_only"]),
            "id_migration": len(classification["id_migration"]),
            "unchanged": len(classification["unchanged"]),
            "stale": len(classification["stale"]),
        },
        "sqlite": sqlite_stats,
    }

    path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Synchronisation incrémentale du RAG LaTeX : "
            "Qdrant dense + SQLite FTS5. "
            "Les chunks inchangés ne sont jamais ré-embeddés."
        )
    )

    ap.add_argument(
        "--processed",
        type=Path,
        default=Path("corpus/processed/current"),
    )
    ap.add_argument(
        "--collection",
        default="ace_ai_crypto_latex_v5",
    )
    ap.add_argument(
        "--model",
        default="bge-m3",
    )
    ap.add_argument(
        "--ollama-host",
        default="http://127.0.0.1:11434",
    )
    ap.add_argument(
        "--qdrant-url",
        default="http://127.0.0.1:6333",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=24,
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher le delta sans modifier Qdrant/SQLite.",
    )
    ap.add_argument(
        "--prune",
        action="store_true",
        help=(
            "Supprimer les points/chunks présents dans l'index "
            "mais absents du corpus courant."
        ),
    )
    ap.add_argument(
        "--sqlite-db",
        type=Path,
        default=None,
        help=(
            "Base lexicale. Par défaut : <processed>/lexical.db"
        ),
    )
    ap.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help=(
            "État de synchronisation JSON. Par défaut : "
            "corpus/index/<collection>_sync_state.json"
        ),
    )

    args = ap.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size doit être >= 1")

    chunks_path = args.processed / "chunks.jsonl"
    relations_path = args.processed / "relations.jsonl"

    chunks = read_jsonl(chunks_path)
    relations = read_jsonl(relations_path)
    current = validate_current_chunks(chunks)

    sqlite_db = (
        args.sqlite_db
        if args.sqlite_db is not None
        else args.processed / "lexical.db"
    )

    state_file = (
        args.state_file
        if args.state_file is not None
        else Path("corpus/index")
        / f"{args.collection}_sync_state.json"
    )

    try:
        import ollama
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise SystemExit(
            "Dépendances manquantes : uv add ollama qdrant-client"
        ) from exc

    qclient = QdrantClient(url=args.qdrant_url)
    collection_exists = qclient.collection_exists(args.collection)

    if collection_exists:
        print(
            f"[SCAN] Qdrant collection={args.collection} "
            "(payloads uniquement, aucun embedding)"
        )
        existing, duplicates = scan_qdrant(
            qclient,
            args.collection,
        )

        if duplicates:
            sample = sorted(set(duplicates))[:20]
            raise RuntimeError(
                "Plusieurs points Qdrant portent le même chunk_id. "
                f"Exemples : {sample}"
            )
    else:
        print(
            f"[INFO] Collection absente : {args.collection}. "
            "Tous les chunks seront considérés comme nouveaux."
        )
        existing = {}

    classification = classify_qdrant(current, existing)

    print("\n=== Incremental RAG Sync Plan ===")
    print(f"Corpus chunks       : {len(current)}")
    print(f"Indexed chunk IDs   : {len(existing)}")
    print(f"NEW                 : {len(classification['new'])}")
    print(
        f"EMBEDDING_CHANGED   : "
        f"{len(classification['embedding_changed'])}"
    )
    print(f"PAYLOAD_ONLY        : {len(classification['payload_only'])}")
    print(f"ID_MIGRATION        : {len(classification['id_migration'])}")
    print(f"UNCHANGED           : {len(classification['unchanged'])}")
    print(f"STALE               : {len(classification['stale'])}")

    embedding_required = (
        len(classification["new"])
        + len(classification["embedding_changed"])
    )

    vector_reuse_required = (
        len(classification["payload_only"])
        + len(classification["id_migration"])
    )

    print(f"Embeddings required : {embedding_required}")
    print(f"Vector reuse only   : {vector_reuse_required}")
    print(
        f"Prune stale         : "
        f"{'YES' if args.prune else 'NO'}"
    )

    if args.dry_run:
        print("\n[DRY-RUN] Aucune modification effectuée.")
        return

    oclient = ollama.Client(host=args.ollama_host)

    # Create collection lazily only if needed.
    if not collection_exists:
        if not chunks:
            raise RuntimeError(
                "Corpus vide : impossible de créer la collection"
            )

        sample_resp = oclient.embed(
            model=args.model,
            input=[chunks[0]["embedding_text"]],
        )
        sample_vec = response_embeddings(sample_resp)[0]

        create_qdrant_collection(
            qclient,
            models,
            args.collection,
            len(sample_vec),
        )

        collection_exists = True
        print(
            f"[OK] Collection créée : {args.collection} "
            f"dim={len(sample_vec)}"
        )

    embed_entries = [
        (chunk, None)
        for chunk in classification["new"]
    ] + [
        (chunk, old)
        for chunk, old in classification["embedding_changed"]
    ]

    if embed_entries:
        embed_and_upsert(
            qclient=qclient,
            models=models,
            oclient=oclient,
            collection=args.collection,
            model_name=args.model,
            entries=embed_entries,
            batch_size=args.batch_size,
        )

    vector_reuse_entries = (
        classification["payload_only"]
        + classification["id_migration"]
    )

    if vector_reuse_entries:
        reuse_vectors_and_upsert(
            qclient=qclient,
            models=models,
            collection=args.collection,
            entries=vector_reuse_entries,
            batch_size=args.batch_size,
        )

    if args.prune and classification["stale"]:
        stale_ids = [
            old["point_id"]
            for _, old in classification["stale"]
        ]

        done = 0
        for batch in batched(stale_ids, 256):
            qclient.delete(
                collection_name=args.collection,
                points_selector=models.PointIdsList(
                    points=batch
                ),
                wait=True,
            )
            done += len(batch)
            print(
                f"Pruned Qdrant {done}/{len(stale_ids)}"
            )

    sqlite_stats = sync_sqlite(
        sqlite_db,
        chunks,
        relations,
        prune=args.prune,
    )

    print(
        f"[OK] SQLite FTS5 synchronized: "
        f"{sqlite_db.resolve()}"
    )
    print(
        "SQLite delta:",
        json.dumps(
            sqlite_stats,
            ensure_ascii=False,
        ),
    )

    info = qclient.get_collection(args.collection)
    qdrant_count = getattr(info, "points_count", None)

    write_sync_state(
        state_file,
        args=args,
        current_chunks=current,
        qdrant_count=qdrant_count,
        classification=classification,
        sqlite_stats=sqlite_stats,
    )

    print(
        f"[OK] Sync state: {state_file.resolve()}"
    )
    print(
        f"[OK] Qdrant collection={args.collection} "
        f"points={qdrant_count} model={args.model}"
    )


if __name__ == "__main__":
    main()
