from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--processed',type=Path,default=Path('corpus/processed/exo7_v1'));ap.add_argument('--max-chars',type=int,default=6000);args=ap.parse_args()
    docs=read(args.processed/'documents.jsonl');units=read(args.processed/'units.jsonl');rels=read(args.processed/'relations.jsonl');chunks=read(args.processed/'chunks.jsonl')
    errors=[];warnings=[]
    def uniq(rs,key,label):
        vals=[r.get(key) for r in rs]
        if any(not x for x in vals):errors.append(f'{label}: {key} manquant')
        d=[x for x,n in Counter(vals).items() if x and n>1]
        if d:errors.append(f'{label}: {len(d)} doublons {key}')
    uniq(docs,'document_id','documents');uniq(units,'unit_id','units');uniq(rels,'relation_id','relations');uniq(chunks,'chunk_id','chunks')
    uid={u['unit_id']:u for u in units};cid={c['chunk_id'] for c in chunks}
    for c in chunks:
        if c['unit_id'] not in uid:errors.append(f"chunk sans unité {c['chunk_id']}")
        if not c.get('latex','').strip():errors.append(f"chunk vide {c['chunk_id']}")
        if len(c.get('latex',''))>args.max_chars:errors.append(f"chunk trop long {c['chunk_id']} {len(c['latex'])}")
    statements={u['unit_id']:u for u in units if u.get('exercise_role')=='statement'}
    for r in rels:
        if r['type'] in {'hint_of','solution_of'}:
            a=uid.get(r['from_unit_id']);b=uid.get(r['to_unit_id'])
            if not a or not b:errors.append(f"relation cassée {r['relation_id']}");continue
            if b.get('exercise_role')!='statement':errors.append(f"cible non statement {r['relation_id']}")
            if a.get('exercise_id')!=b.get('exercise_id'):errors.append(f"exercise_id incohérent {r['relation_id']}")
    refs=[r for r in rels if r['type']=='references'];unres=[r for r in refs if not r.get('resolved')]
    anomalies=[a for u in units for a in u.get('source_anomalies',[]) if a]
    if anomalies:warnings.append(f'anomalies source conservées: {Counter(anomalies)}')
    if unres:warnings.append(f'références internes non résolues: {len(unres)}/{len(refs)}')
    print('=== Exo7 Exercise RAG Validation ===')
    print('documents:',len(docs),'units:',len(units),'chunks:',len(chunks))
    print('statements:',len(statements),'hints:',sum(u.get('exercise_role')=='hint' for u in units),'solutions:',sum(u.get('exercise_role')=='solution' for u in units))
    print('references:',len(refs),'unresolved:',len(unres),'max_chunk_chars:',max((len(c['latex']) for c in chunks),default=0))
    print('errors:',len(errors),'warnings:',len(warnings))
    for w in warnings:print('[WARN]',w)
    for e in errors[:50]:print('[ERROR]',e)
    if errors:raise SystemExit(1)
    print('[OK] Exercise dataset structurally valid')
if __name__=='__main__':main()
