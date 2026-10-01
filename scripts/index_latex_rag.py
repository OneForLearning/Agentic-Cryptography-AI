from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from latex_rag_core import read_jsonl


DEFAULT_COLLECTION = "ace_ai_crypto_latex_v1"


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def response_embeddings(resp):
    if isinstance(resp, dict):
        value = resp.get("embeddings") or resp.get("embedding")
        if value is not None:
            return value
    if hasattr(resp, "embeddings"):
        return resp.embeddings
    if hasattr(resp, "embedding"):
        return [resp.embedding]
    raise RuntimeError("Unknown Ollama embedding response")


def build_sqlite(db_path, chunks, relations):
    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(
        db_path
    )

    try:
        con.execute(
            "PRAGMA journal_mode=WAL"
        )

        con.execute(
            """
            CREATE TABLE chunks (
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
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
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
            CREATE TABLE relations (
                type TEXT,
                from_unit_id TEXT,
                to_unit_id TEXT,
                target_label TEXT,
                resolved INTEGER
            )
            """
        )

        for chunk in chunks:
            payload = json.dumps(
                chunk,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            con.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                (
                    chunk["chunk_id"],
                    chunk["unit_id"],
                    chunk["document_id"],
                    chunk["unit_type"],
                    payload,
                    chunk["embedding_text"],
                ),
            )

            con.execute(
                "INSERT INTO chunks_fts VALUES (?,?,?,?)",
                (
                    chunk["chunk_id"],
                    chunk["document_id"],
                    chunk["unit_type"],
                    chunk["embedding_text"],
                ),
            )

        for relation in relations:
            con.execute(
                "INSERT INTO relations VALUES (?,?,?,?,?)",
                (
                    relation.get("type"),
                    relation.get("from_unit_id"),
                    relation.get("to_unit_id"),
                    relation.get("target_label"),
                    1
                    if relation.get("resolved", True)
                    else 0,
                ),
            )

        con.execute(
            "CREATE INDEX idx_chunks_unit ON chunks(unit_id)"
        )
        con.execute(
            "CREATE INDEX idx_rel_from ON relations(from_unit_id)"
        )
        con.execute(
            "CREATE INDEX idx_rel_to ON relations(to_unit_id)"
        )

        con.commit()

    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Embed LaTeX chunks with Ollama bge-m3 and index in "
            "Qdrant + SQLite FTS5."
        )
    )

    ap.add_argument(
        "--processed",
        type=Path,
        default=Path("corpus/processed/latex"),
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
        "--collection",
        default=DEFAULT_COLLECTION,
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=24,
    )
    ap.add_argument(
        "--recreate",
        action="store_true",
    )

    args = ap.parse_args()

    try:
        import ollama
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies: uv add ollama qdrant-client"
        ) from exc

    chunks = read_jsonl(
        args.processed / "chunks.jsonl"
    )
    relations = read_jsonl(
        args.processed / "relations.jsonl"
    )

    if not chunks:
        raise SystemExit(
            "No chunks"
        )

    build_sqlite(
        args.processed / "lexical.db",
        chunks,
        relations,
    )

    print(
        f"[OK] SQLite FTS5: "
        f"{(args.processed / 'lexical.db').resolve()}"
    )

    oclient = ollama.Client(
        host=args.ollama_host
    )

    first_resp = oclient.embed(
        model=args.model,
        input=[chunks[0]["embedding_text"]],
    )

    first_vec = response_embeddings(
        first_resp
    )[0]

    dim = len(
        first_vec
    )

    print(
        "Embedding dimension:",
        dim,
    )

    qclient = QdrantClient(
        url=args.qdrant_url
    )

    exists = qclient.collection_exists(
        args.collection
    )

    if exists and args.recreate:
        qclient.delete_collection(
            args.collection
        )
        exists = False

    if not exists:
        qclient.create_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        )

    for field in [
        "document_id",
        "unit_type",
        "language",
        "source_file",
        "label",
        "content_hash",
    ]:
        try:
            qclient.create_payload_index(
                args.collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    total = len(
        chunks
    )

    done = 0

    for batch in batched(
        chunks,
        args.batch_size,
    ):
        texts = [
            chunk["embedding_text"]
            for chunk in batch
        ]

        response = oclient.embed(
            model=args.model,
            input=texts,
        )

        vectors = response_embeddings(
            response
        )

        if len(vectors) != len(batch):
            raise RuntimeError(
                "Embedding batch size mismatch"
            )

        points = []

        for chunk, vector in zip(
            batch,
            vectors,
        ):
            payload = {
                key: value
                for key, value in chunk.items()
                if key
                not in {
                    "embedding_text",
                    "latex",
                }
            }

            payload["embedding_text"] = chunk[
                "embedding_text"
            ]
            payload["latex"] = chunk[
                "latex"
            ]

            points.append(
                models.PointStruct(
                    id=chunk["qdrant_id"],
                    vector=vector,
                    payload=payload,
                )
            )

        qclient.upsert(
            collection_name=args.collection,
            points=points,
            wait=True,
        )

        done += len(
            batch
        )

        print(
            f"Indexed {done}/{total}"
        )

    info = qclient.get_collection(
        args.collection
    )

    print(
        f"[OK] Qdrant collection={args.collection} "
        f"points={getattr(info, 'points_count', None)} "
        f"model={args.model}"
    )


if __name__ == "__main__":
    main()
