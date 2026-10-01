from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter,defaultdict
from latex_rag_core import read_jsonl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--processed',type=Path,default=Path('corpus/processed/latex'))
    ap.add_argument('--strict',action='store_true')
    args=ap.parse_args()
    docs=read_jsonl(args.processed/'documents.jsonl')
    units=read_jsonl(args.processed/'units.jsonl')
    rels=read_jsonl(args.processed/'relations.jsonl')
    chunks=read_jsonl(args.processed/'chunks.jsonl')
    errors=[];warnings=[]
    def duplicates(items,key,label):
        seen={}
        for x in items:
            v=x.get(key)
            if not v: errors.append(f'{label}: missing {key}');continue
            if v in seen: errors.append(f'{label}: duplicate {key}={v}')
            seen[v]=1
    duplicates(docs,'document_id','documents');duplicates(units,'unit_id','units');duplicates(chunks,'chunk_id','chunks')
    unit_ids={u['unit_id'] for u in units}
    for u in units:
        if u.get('proof_of') and u['proof_of'] not in unit_ids: errors.append(f"proof target missing: {u['unit_id']} -> {u['proof_of']}")
    for c in chunks:
        if c['unit_id'] not in unit_ids: errors.append(f"chunk unit missing: {c['chunk_id']}")
        if not c.get('embedding_text','').strip(): errors.append(f"empty embedding text: {c['chunk_id']}")
        if not c.get('latex','').strip(): errors.append(f"empty latex: {c['chunk_id']}")
    ref=[r for r in rels if r['type']=='references']
    unresolved=[r for r in ref if not r.get('resolved')]
    if ref and len(unresolved)/len(ref)>0.02: warnings.append(f'unresolved refs {len(unresolved)}/{len(ref)} >2%')
    incomplete=[d for d in docs if not d.get('complete_inputs')]
    if incomplete: warnings.append(f'{len(incomplete)} documents have missing \\input/\\include targets')
    macro_inc=[d for d in docs if not d.get('macro_context_complete')]
    if macro_inc: warnings.append(f'{len(macro_inc)} documents have incomplete macro context')
    proof=[u for u in units if u['unit_type']=='proof']
    linked=[u for u in proof if u.get('proof_of')]
    if proof and len(linked)/len(proof)<0.90: warnings.append(f'proof association only {len(linked)}/{len(proof)}')
    print('=== LaTeX RAG Validation ===')
    print('documents:',len(docs),'units:',len(units),'chunks:',len(chunks))
    print('references:',len(ref),'unresolved:',len(unresolved))
    print('proofs:',len(proof),'linked:',len(linked))
    print('errors:',len(errors),'warnings:',len(warnings))
    for w in warnings: print('[WARN]',w)
    for e in errors: print('[ERROR]',e)
    if errors or (args.strict and warnings): raise SystemExit(1)
    print('[OK] Structural dataset valid')

if __name__=='__main__': main()
