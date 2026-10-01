from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter
from latex_rag_core import read_jsonl, build_chunks, write_jsonl

def main():
    ap=argparse.ArgumentParser(description='Build structure-aware RAG chunks from extracted LaTeX units.')
    ap.add_argument('--processed', type=Path, default=Path('corpus/processed/latex'))
    ap.add_argument('--max-chars',type=int,default=6000)
    ap.add_argument('--target-chars',type=int,default=4500)
    ap.add_argument('--prose-overlap',type=int,default=500)
    args=ap.parse_args()
    docs=read_jsonl(args.processed/'documents.jsonl')
    units=read_jsonl(args.processed/'units.jsonl')
    macros=read_jsonl(args.processed/'macros.jsonl')
    chunks=build_chunks(docs,units,macros,max_chars=args.max_chars,target_chars=args.target_chars,prose_overlap_chars=args.prose_overlap)
    write_jsonl(args.processed/'chunks.jsonl',chunks)
    stats={
        'chunks':len(chunks),
        'unit_types':dict(Counter(c['unit_type'] for c in chunks)),
        'documents':len(set(c['document_id'] for c in chunks)),
        'max_embedding_chars':max((len(c['embedding_text']) for c in chunks),default=0),
        'max_latex_chars':max((len(c['latex']) for c in chunks),default=0),
        'proof_chunks':sum(c['unit_type']=='proof' for c in chunks),
        'proof_chunks_linked':sum(c['unit_type']=='proof' and bool(c.get('proof_of')) for c in chunks),
    }
    (args.processed/'chunk_stats.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(stats,indent=2,ensure_ascii=False))
    print(f'[OK] {(args.processed/"chunks.jsonl").resolve()}')

if __name__=='__main__': main()
