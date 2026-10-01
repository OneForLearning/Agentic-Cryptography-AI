from __future__ import annotations
import argparse, json
from pathlib import Path

def read(path):
    out=[]
    with path.open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if line.strip():
                out.append(json.loads(line))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed",type=Path,required=True)
    args=ap.parse_args()
    p=args.processed
    docs=read(p/"documents.jsonl")
    units=read(p/"units.jsonl")
    rels=read(p/"relations.jsonl")
    chunks=read(p/"chunks.jsonl")
    envs=read(p/"declared_environments.jsonl")
    stats=json.loads((p/"stats.json").read_text(encoding="utf-8"))
    errors=[]; warnings=[]
    if len(chunks)!=stats.get("outline_chunks"): errors.append("chunk count != stats")
    if any(c.get("content_kind")!="outline" for c in chunks): errors.append("non-outline chunk found")
    if any(not c.get("exclude_from_answer_generation") for c in chunks): errors.append("outline chunk not excluded from answer generation")
    if any(c.get("knowledge_status")!="placeholder_only" for c in chunks): errors.append("unexpected knowledge_status")
    if stats.get("knowledge_chunks")!=0: errors.append("knowledge_chunks should be 0")
    used=sum(e.get("usage_count",0) for e in envs)
    if used!=0: warnings.append(f"formal environments are now used: {used}; rebuild policy may need upgrading")
    if stats.get("placeholder_comments",0)>0:
        warnings.append(f"source contains {stats['placeholder_comments']} placeholder comments")
    print("=== Serious Outline RAG Validation ===")
    print(f"documents: {len(docs)} units: {len(units)} chunks: {len(chunks)} relations: {len(rels)}")
    print(f"knowledge_chunks: {stats.get('knowledge_chunks')} formal_env_usage: {stats.get('formal_environment_usage')}")
    print(f"errors: {len(errors)} warnings: {len(warnings)}")
    for x in errors: print("[ERROR]",x)
    for x in warnings: print("[WARN]",x)
    if errors: raise SystemExit(1)
    print("[OK] Navigation-only dataset structurally valid")
    print("[DO NOT MERGE] This source is outline-only and contains no factual mathematical content.")

if __name__=="__main__":
    main()
