from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter
from latex_rag_core import (
    build_document_graph, parse_macros, formal_units_for_file, pair_proofs,
    build_relations, write_jsonl, normalize_unicode
)

def main():
    ap=argparse.ArgumentParser(description='Extract structured LaTeX documents, macros, units and relations.')
    ap.add_argument('--latex-root', type=Path, default=Path('corpus/raw/latex/expanded'))
    ap.add_argument('--out', type=Path, default=Path('corpus/processed/latex'))
    ap.add_argument('--proof-max-gap', type=int, default=3500)
    args=ap.parse_args()
    latex_root=args.latex_root.resolve()
    docs, components=build_document_graph(latex_root)
    all_macros=[]
    all_units=[]
    for doc in docs:
        doc_id=doc['document_id']
        macro_records=[]
        for p in components[doc_id]:
            text=normalize_unicode(p.read_text(encoding='utf-8',errors='replace'))
            macro_records.extend(parse_macros(text,str(p.relative_to(latex_root)),doc_id))
        # Last definition wins within document; retain all records in macros.jsonl for provenance.
        macro_names={m['command'] for m in macro_records}
        all_macros.extend(macro_records)
        for p in components[doc_id]:
            all_units.extend(formal_units_for_file(doc,p,latex_root,macro_names))
    pair_proofs(all_units,args.proof_max_gap)
    relations, rel_stats=build_relations(docs,all_units)
    args.out.mkdir(parents=True,exist_ok=True)
    write_jsonl(args.out/'documents.jsonl',docs)
    write_jsonl(args.out/'macros.jsonl',all_macros)
    write_jsonl(args.out/'units.jsonl',all_units)
    write_jsonl(args.out/'relations.jsonl',relations)
    types=Counter(u['unit_type'] for u in all_units)
    stats={
        'documents':len(docs),'component_files':sum(len(x) for x in components.values()),
        'macros':len(all_macros),'units':len(all_units),'unit_types':dict(types),
        'incomplete_documents':sum(not d['complete_inputs'] for d in docs),
        'macro_context_incomplete_documents':sum(not d['macro_context_complete'] for d in docs),
        **rel_stats,
    }
    (args.out/'extraction_stats.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(stats,indent=2,ensure_ascii=False))
    print(f'[OK] {args.out.resolve()}')

if __name__=='__main__': main()
