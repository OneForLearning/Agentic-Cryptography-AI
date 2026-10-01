from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


DEFAULT_COLLECTION = "ace_ai_crypto_latex_v1"


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


def rrf(rankings, k=60):
    score = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            score[cid] += 1.0 / (k + rank)
    return score


def fts_query(q):
    terms = re.findall(r"[\wÀ-ÿ]+", q, flags=re.UNICODE)
    cleaned = [
        '"' + t.replace('"', '') + '"'
        for t in terms
        if len(t) > 1
    ]
    return " OR ".join(cleaned) or '"' + q.replace('"', '') + '"'


def qdrant_hits(client, collection, vector, limit):
    if not client.collection_exists(collection):
        available = [
            item.name
            for item in client.get_collections().collections
        ]
        raise RuntimeError(
            "Collection Qdrant introuvable : "
            f"{collection!r}. Collections disponibles : {available}. "
            "Lance d'abord index_latex_rag.py ou passe --collection avec "
            "le nom exact de la collection existante."
        )

    result = client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        with_payload=True,
    )

    points = getattr(result, "points", result)

    out = []

    for point in points:
        payload = getattr(point, "payload", {}) or {}
        cid = payload.get("chunk_id")

        if cid:
            out.append(
                (
                    cid,
                    float(getattr(point, "score", 0.0)),
                    payload,
                )
            )

    return out


def type_bonus(query, unit_type):
    q = query.lower()

    if any(w in q for w in ["démontr", "preuve", "prove", "proof"]):
        if unit_type == "proof":
            return 0.12

    if any(
        w in q
        for w in [
            "défin",
            "definition",
            "what is",
            "qu’est-ce",
            "qu'est-ce",
        ]
    ):
        if unit_type == "definition":
            return 0.12

    if any(w in q for w in ["théorème", "theorem"]):
        if unit_type == "theorem":
            return 0.08

    if any(w in q for w in ["lemme", "lemma"]):
        if unit_type == "lemma":
            return 0.08

    if any(w in q for w in ["corollaire", "corollary"]):
        if unit_type == "corollary":
            return 0.08

    return 0.0


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Hybrid dense+FTS retrieval with theorem/proof graph expansion."
        )
    )

    ap.add_argument("query")
    ap.add_argument(
        "--processed",
        type=Path,
        default=Path("corpus/processed/latex"),
    )
    ap.add_argument("--model", default="bge-m3")
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
    ap.add_argument("--dense-k", type=int, default=24)
    ap.add_argument("--lexical-k", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument(
        "--max-context-chars",
        type=int,
        default=30000,
    )
    ap.add_argument("--json", action="store_true")

    args = ap.parse_args()

    lexical_db = args.processed / "lexical.db"

    if not lexical_db.exists():
        raise SystemExit(
            f"Base lexicale absente : {lexical_db}. "
            "Lance d'abord scripts/index_latex_rag.py."
        )

    try:
        import ollama
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise SystemExit(
            "Dépendances manquantes : uv add ollama qdrant-client"
        ) from exc

    oclient = ollama.Client(
        host=args.ollama_host
    )

    qclient = QdrantClient(
        url=args.qdrant_url
    )

    embed_resp = oclient.embed(
        model=args.model,
        input=[args.query],
    )

    vector = response_embeddings(
        embed_resp
    )[0]

    dense = qdrant_hits(
        qclient,
        args.collection,
        vector,
        args.dense_k,
    )

    con = sqlite3.connect(
        lexical_db
    )

    con.row_factory = sqlite3.Row

    try:
        try:
            rows = con.execute(
                """
                SELECT chunk_id,
                       bm25(chunks_fts) AS s
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY s
                LIMIT ?
                """,
                (
                    fts_query(args.query),
                    args.lexical_k,
                ),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

        lexical = [
            row["chunk_id"]
            for row in rows
        ]

        dense_ids = [
            item[0]
            for item in dense
        ]

        scores = rrf(
            [
                dense_ids,
                lexical,
            ]
        )

        candidate_ids = set(
            scores
        )

        payloads = {}

        for cid in candidate_ids:
            row = con.execute(
                """
                SELECT payload_json
                FROM chunks
                WHERE chunk_id=?
                """,
                (cid,),
            ).fetchone()

            if row:
                payloads[cid] = json.loads(
                    row[0]
                )

        for cid, payload in payloads.items():
            scores[cid] += type_bonus(
                args.query,
                payload.get("unit_type"),
            )

        ranked = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        selected = []
        doc_counts = defaultdict(int)
        seen_hash = set()
        chars = 0

        def add_chunk(cid, reason):
            nonlocal chars

            if any(
                item["chunk_id"] == cid
                for item in selected
            ):
                return

            row = con.execute(
                """
                SELECT payload_json
                FROM chunks
                WHERE chunk_id=?
                """,
                (cid,),
            ).fetchone()

            if not row:
                return

            payload = json.loads(
                row[0]
            )

            content_hash = payload.get(
                "content_hash"
            )

            if content_hash in seen_hash:
                return

            if (
                doc_counts[
                    payload["document_id"]
                ] >= 4
                and reason == "base"
            ):
                return

            cost = len(
                payload.get("latex", "")
            )

            if (
                chars + cost
                > args.max_context_chars
                and selected
            ):
                return

            payload["_retrieval_reason"] = reason
            payload["_fusion_score"] = scores.get(
                cid,
                0.0,
            )

            selected.append(
                payload
            )

            if content_hash:
                seen_hash.add(
                    content_hash
                )

            doc_counts[
                payload["document_id"]
            ] += 1

            chars += cost

        for cid in ranked[: args.top_k]:
            add_chunk(
                cid,
                "base",
            )

            payload = next(
                (
                    item
                    for item in selected
                    if item["chunk_id"] == cid
                ),
                None,
            )

            if not payload:
                continue

            unit_id = payload["unit_id"]

            related_units = set(
                payload.get(
                    "related_unit_ids",
                    [],
                )
            )

            if payload.get("proof_of"):
                related_units.add(
                    payload["proof_of"]
                )

            relrows = con.execute(
                """
                SELECT to_unit_id
                FROM relations
                WHERE from_unit_id=?
                  AND type IN ('proof_of', 'references')
                  AND to_unit_id IS NOT NULL
                """,
                (unit_id,),
            ).fetchall()

            related_units.update(
                row[0]
                for row in relrows
            )

            back = con.execute(
                """
                SELECT from_unit_id
                FROM relations
                WHERE to_unit_id=?
                  AND type='proof_of'
                """,
                (unit_id,),
            ).fetchall()

            related_units.update(
                row[0]
                for row in back
            )

            for related_unit in list(
                related_units
            )[:4]:
                chunk_rows = con.execute(
                    """
                    SELECT chunk_id
                    FROM chunks
                    WHERE unit_id=?
                    ORDER BY chunk_id
                    LIMIT 2
                    """,
                    (related_unit,),
                ).fetchall()

                for row in chunk_rows:
                    add_chunk(
                        row[0],
                        "structural",
                    )

        if args.json:
            print(
                json.dumps(
                    selected,
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(
                f"Collection: {args.collection}"
            )
            print(
                f"Query: {args.query}"
            )
            print(
                f"Results: {len(selected)}\n"
            )

            for index, payload in enumerate(
                selected,
                1,
            ):
                print("=" * 88)
                print(
                    f"[{index}] "
                    f"{payload['chunk_id']} "
                    f"score={payload.get('_fusion_score', 0):.5f} "
                    f"reason={payload['_retrieval_reason']}"
                )
                print(
                    "Course:",
                    payload.get("course_title"),
                )
                print(
                    "Section:",
                    payload.get("section_path_text"),
                )
                print(
                    "Type:",
                    payload.get("unit_type"),
                    "Label:",
                    payload.get("label"),
                    "Proof of:",
                    payload.get("proof_of"),
                )
                print("-" * 88)
                print(
                    payload.get(
                        "latex",
                        "",
                    )[:5000]
                )

    finally:
        con.close()


if __name__ == "__main__":
    main()
