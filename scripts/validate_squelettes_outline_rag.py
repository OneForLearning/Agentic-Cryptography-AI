
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter

def read_jsonl(p):
    if not p.exists(): raise FileNotFoundError(p)
    rows=[]
    with p.open(encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try: rows.append(json.loads(line))
                except Exception as e: raise RuntimeError(f"{p}:{i}: {e}")
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed",type=Path,required=True)
    a=ap.parse_args(); p=a.processed
    docs=read_jsonl(p/"documents.jsonl")
    units=read_jsonl(p/"units.jsonl")
    chunks=read_jsonl(p/"chunks.jsonl")
    rels=read_jsonl(p/"relations.jsonl")
    flags=read_jsonl(p/"quality_flags.jsonl")
    stats=json.loads((p/"stats.json").read_text(encoding="utf-8"))
    errors=[]; warnings=[]
    def uniq(rows,key,label):
        vals=[r.get(key) for r in rows]
        if any(v is None for v in vals): errors.append(f"{label}: missing {key}")
        if len(vals)!=len(set(vals)): errors.append(f"{label}: duplicate {key}")
    uniq(docs,"document_id","documents"); uniq(units,"unit_id","units"); uniq(chunks,"chunk_id","chunks"); uniq(rels,"relation_id","relations")
    did={d["document_id"] for d in docs}; uid={u["unit_id"] for u in units}
    for u in units:
        if u.get("document_id") not in did: errors.append(f"unit without document: {u.get('unit_id')}")
    for c in chunks:
        if c.get("unit_id") not in uid: errors.append(f"chunk without unit: {c.get('chunk_id')}")
        if not c.get("exclude_from_answer_generation",False): errors.append(f"knowledge chunk unexpectedly enabled: {c.get('chunk_id')}")
        if c.get("content_kind")!="outline": errors.append(f"non-outline chunk: {c.get('chunk_id')}")
    for r in rels:
        if r.get("from_unit_id") not in uid or r.get("to_unit_id") not in uid:
            errors.append(f"dangling relation: {r.get('relation_id')}")
    if stats.get("knowledge_chunks")!=0: errors.append("stats knowledge_chunks != 0")
    if stats.get("content_display_math_blocks")!=0:
        warnings.append(f"source contains {stats['content_display_math_blocks']} display-math blocks; inspect manually")
    if flags:
        warnings.append(f"quality flags: {Counter(f['type'] for f in flags)}")
    warnings.append("source declares rag_indexing_policy=outline_only; these chunks are routing metadata, not mathematical knowledge")
    print("=== Squelettes Outline RAG Validation ===")
    print(f"documents: {len(docs)} units: {len(units)} chunks: {len(chunks)} relations: {len(rels)}")
    print(f"knowledge_chunks: {stats.get('knowledge_chunks')} formal_placeholders: {stats.get('formal_placeholders')}")
    print(f"display_math_blocks: {stats.get('content_display_math_blocks')} refs: {stats.get('content_refs')}")
    print(f"errors: {len(errors)} warnings: {len(warnings)}")
    for x in errors[:20]: print("[ERROR]",x)
    for x in warnings: print("[WARN]",x)
    if errors: raise SystemExit(1)
    print("[OK] Navigation-only mathematical skeleton dataset structurally valid")
    print("[DO NOT MERGE] Do not use these chunks as factual mathematical evidence.")
if __name__=="__main__": main()
