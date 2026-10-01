from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

DEFAULT_COLLECTION = "ace_ai_crypto_latex_v3"


def response_embeddings(resp):
    if isinstance(resp, dict):
        value = resp.get("embeddings") or resp.get("embedding")
        if value is not None:
            return value
    if hasattr(resp, "embeddings"):
        return resp.embeddings
    if hasattr(resp, "embedding"):
        return [resp.embedding]
    raise RuntimeError("Réponse d'embedding Ollama inconnue")


def rrf(rankings, k=60):
    score = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            score[cid] += 1.0 / (k + rank)
    return score


def fts_query(q):
    terms = re.findall(r"[\wÀ-ÿ]+", q, flags=re.UNICODE)
    cleaned = ['"' + t.replace('"', '') + '"' for t in terms if len(t) > 1]
    return " OR ".join(cleaned) or '"' + q.replace('"', '') + '"'


def qdrant_hits(client, collection, vector, limit):
    if not client.collection_exists(collection):
        available = [x.name for x in client.get_collections().collections]
        raise RuntimeError(f"Collection Qdrant introuvable : {collection!r}. Disponibles : {available}")
    res = client.query_points(collection_name=collection, query=vector, limit=limit, with_payload=True)
    points = getattr(res, "points", res)
    out = []
    for p in points:
        payload = getattr(p, "payload", {}) or {}
        cid = payload.get("chunk_id")
        if cid:
            out.append((cid, float(getattr(p, "score", 0.0)), payload))
    return out


def type_bonus(query: str, unit_type: str | None) -> float:
    q = query.lower()
    bonus = 0.0
    if any(w in q for w in ["démontr", "preuve", "prove", "proof"]):
        if unit_type == "proof": bonus += 0.12
    if any(w in q for w in ["défin", "definition", "qu'est-ce", "qu’est-ce"]):
        if unit_type == "definition": bonus += 0.12
    if "théorème" in q or "theorem" in q:
        if unit_type == "theorem": bonus += 0.08
    if any(w in q for w in ["exercice", "exercise", "problème", "problem"]):
        if unit_type == "exercise_statement": bonus += 0.10
    if any(w in q for w in ["corrig", "correction", "solution", "résous", "résoudre"]):
        if unit_type == "exercise_solution": bonus += 0.13
    if any(w in q for w in ["indication", "indice", "hint"]):
        if unit_type == "exercise_hint": bonus += 0.13
    return bonus


def main():
    ap = argparse.ArgumentParser(description="Hybrid LaTeX retrieval with theorem/proof and exercise statement/hint/solution expansion.")
    ap.add_argument("query")
    ap.add_argument("--processed", type=Path, default=Path("corpus/processed/latex_v3"))
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--dense-k", type=int, default=30)
    ap.add_argument("--lexical-k", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--max-context-chars", type=int, default=32000)
    ap.add_argument("--exercise-mode", choices=["auto","statement","hint","solution","full"], default="auto", help="Controls structural expansion of exercise statement/hint/solution.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    qlow = args.query.lower()
    exercise_mode = args.exercise_mode
    if exercise_mode == "auto":
        if any(w in qlow for w in ["corrig", "correction", "solution", "résous", "résoudre"]):
            exercise_mode = "full"
        elif any(w in qlow for w in ["indication", "indice", "hint"]):
            exercise_mode = "hint"
        elif any(w in qlow for w in ["exercice", "exercise", "problème", "problem"]):
            exercise_mode = "statement"
        else:
            exercise_mode = "statement"

    db = args.processed / "lexical.db"
    if not db.exists():
        raise SystemExit(f"Base lexicale absente : {db}. Lance d'abord index_latex_rag.py.")

    import ollama
    from qdrant_client import QdrantClient

    oclient = ollama.Client(host=args.ollama_host)
    qclient = QdrantClient(url=args.qdrant_url)
    vector = response_embeddings(oclient.embed(model=args.model, input=[args.query]))[0]
    dense = qdrant_hits(qclient, args.collection, vector, args.dense_k)

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        try:
            rows = con.execute(
                "SELECT chunk_id,bm25(chunks_fts) AS s FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?",
                (fts_query(args.query), args.lexical_k),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        lexical = [r["chunk_id"] for r in rows]
        scores = rrf([[x[0] for x in dense], lexical])
        payloads = {}
        for cid in scores:
            row = con.execute("SELECT payload_json FROM chunks WHERE chunk_id=?", (cid,)).fetchone()
            if row:
                payloads[cid] = json.loads(row[0])
        for cid, payload in payloads.items():
            scores[cid] += type_bonus(args.query, payload.get("unit_type"))
        ranked = sorted(scores, key=scores.get, reverse=True)

        selected = []
        selected_ids = set()
        seen_hash = set()
        doc_counts = defaultdict(int)
        chars = 0

        def add_chunk(cid, reason):
            nonlocal chars
            if cid in selected_ids:
                return
            row = con.execute("SELECT payload_json FROM chunks WHERE chunk_id=?", (cid,)).fetchone()
            if not row:
                return
            p = json.loads(row[0])
            h = p.get("content_hash")
            if h and h in seen_hash:
                return
            if reason == "base" and doc_counts[p["document_id"]] >= 5:
                return
            cost = len(p.get("latex", ""))
            if chars + cost > args.max_context_chars and selected:
                return
            p["_retrieval_reason"] = reason
            p["_fusion_score"] = scores.get(cid, 0.0)
            selected.append(p)
            selected_ids.add(cid)
            if h:
                seen_hash.add(h)
            doc_counts[p["document_id"]] += 1
            chars += cost

        def chunks_for_unit(unit_id):
            rows = con.execute("SELECT chunk_id,payload_json FROM chunks WHERE unit_id=?", (unit_id,)).fetchall()
            parsed = [(r["chunk_id"], json.loads(r["payload_json"])) for r in rows]
            parsed.sort(key=lambda x: (x[1].get("part", 1), x[0]))
            return [x[0] for x in parsed]

        for cid in ranked[: args.top_k]:
            add_chunk(cid, "base")
            row = con.execute("SELECT payload_json FROM chunks WHERE chunk_id=?", (cid,)).fetchone()
            if not row:
                continue
            p = json.loads(row[0])
            unit_id = p["unit_id"]
            related = set()

            if p.get("content_kind") == "exercise":
                # A selected hint/solution always gets its statement parent.
                forward_types = ["references"]
                if p.get("exercise_role") == "hint":
                    forward_types.append("hint_of")
                elif p.get("exercise_role") == "solution":
                    forward_types.append("solution_of")
                placeholders = ",".join("?" for _ in forward_types)
                forward = con.execute(
                    f"SELECT to_unit_id FROM relations WHERE from_unit_id=? AND type IN ({placeholders}) AND to_unit_id IS NOT NULL",
                    (unit_id, *forward_types),
                ).fetchall()
                related.update(r[0] for r in forward)

                # Children are added only according to the pedagogical mode.
                child_types = []
                if exercise_mode in {"hint", "full"}:
                    child_types.append("hint_of")
                if exercise_mode in {"solution", "full"}:
                    child_types.append("solution_of")
                if child_types:
                    placeholders = ",".join("?" for _ in child_types)
                    backward = con.execute(
                        f"SELECT from_unit_id FROM relations WHERE to_unit_id=? AND type IN ({placeholders})",
                        (unit_id, *child_types),
                    ).fetchall()
                    related.update(r[0] for r in backward)
            else:
                related.update(p.get("related_unit_ids", []))
                if p.get("proof_of"):
                    related.add(p["proof_of"])
                forward = con.execute(
                    "SELECT to_unit_id FROM relations WHERE from_unit_id=? AND type IN ('proof_of','references') AND to_unit_id IS NOT NULL",
                    (unit_id,),
                ).fetchall()
                backward = con.execute(
                    "SELECT from_unit_id FROM relations WHERE to_unit_id=? AND type='proof_of'",
                    (unit_id,),
                ).fetchall()
                related.update(r[0] for r in forward)
                related.update(r[0] for r in backward)

            for rid in related:
                for rcid in chunks_for_unit(rid):
                    add_chunk(rcid, "structural")

        if args.json:
            print(json.dumps(selected, indent=2, ensure_ascii=False))
        else:
            print(f"Collection: {args.collection}")
            print(f"Query: {args.query}")
            print(f"Exercise mode: {exercise_mode}")
            print(f"Results: {len(selected)}\n")
            for i,p in enumerate(selected,1):
                print("="*88)
                print(f"[{i}] {p['chunk_id']} score={p.get('_fusion_score',0):.5f} reason={p['_retrieval_reason']}")
                print("Course:",p.get("course_title"))
                print("Section:",p.get("section_path_text"))
                print("Type:",p.get("unit_type"),"Exercise:",p.get("exercise_id"),"Role:",p.get("exercise_role"))
                print("-"*88)
                print(p.get("latex","")[:6000])
    finally:
        con.close()

if __name__ == "__main__":
    main()
