from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path

REQ=['documents.jsonl','macros.jsonl','units.jsonl','chunks.jsonl','relations.jsonl','formulas.jsonl','formula_relations.jsonl','formal_objects.jsonl','topics.jsonl','exercise_topics.jsonl','semantic_annotations.jsonl','related_clusters.jsonl','duplicate_groups.jsonl','quality_flags.jsonl','unresolved_refs.jsonl','stats.json']

def readj(path):
    out=[]
    with path.open(encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try:r=json.loads(line)
            except Exception as e: raise RuntimeError(f'{path}:{n}: {e}')
            out.append(r)
    return out

def uniq(records,key,label,errors):
    seen=set()
    for r in records:
        v=r.get(key)
        if not v: errors.append(f'{label}: missing {key}')
        elif v in seen: errors.append(f'{label}: duplicate {key}={v}')
        else: seen.add(v)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--processed',type=Path,required=True);args=ap.parse_args()
    p=args.processed;errors=[];warnings=[]
    for n in REQ:
        if not (p/n).exists(): errors.append(f'missing file: {n}')
    if errors:
        print('errors:',len(errors));[print('[ERROR]',x) for x in errors];raise SystemExit(1)
    docs=readj(p/'documents.jsonl');units=readj(p/'units.jsonl');chunks=readj(p/'chunks.jsonl');rels=readj(p/'relations.jsonl');forms=readj(p/'formulas.jsonl');frels=readj(p/'formula_relations.jsonl');formal=readj(p/'formal_objects.jsonl');topics=readj(p/'topics.jsonl');clusters=readj(p/'related_clusters.jsonl');qf=readj(p/'quality_flags.jsonl');unres=readj(p/'unresolved_refs.jsonl')
    uniq(docs,'document_id','documents',errors);uniq(units,'unit_id','units',errors);uniq(chunks,'chunk_id','chunks',errors);uniq(rels,'relation_id','relations',errors);uniq(forms,'formula_id','formulas',errors);uniq(formal,'formal_id','formal_objects',errors);uniq(topics,'topic_id','topics',errors);uniq(clusters,'related_cluster_id','related_clusters',errors)
    did={d['document_id'] for d in docs};uids={u['unit_id'] for u in units};cids={c['chunk_id'] for c in chunks};fids={f['formula_id'] for f in forms};eids={u['exercise_id'] for u in units if u['role']=='statement'}
    for u in units:
        if u.get('document_id') not in did: errors.append(f"unit {u.get('unit_id')} has missing document")
    for c in chunks:
        if c.get('unit_id') not in uids: errors.append(f"chunk {c.get('chunk_id')} has missing unit")
        if len(c.get('latex',''))>6000: errors.append(f"chunk too long {c.get('chunk_id')}: {len(c.get('latex',''))}")
        if not c.get('embedding_text'): errors.append(f"empty embedding_text {c.get('chunk_id')}")
    for r in rels:
        if r.get('from_unit_id') and r['from_unit_id'] not in uids: errors.append(f"relation {r['relation_id']} missing from unit")
        if r.get('resolved') and r.get('to_unit_id') and r['to_unit_id'] not in uids: errors.append(f"relation {r['relation_id']} missing to unit")
    for f in forms:
        if f.get('chunk_id') not in cids: errors.append(f"formula {f['formula_id']} missing chunk")
        if f.get('unit_id') not in uids: errors.append(f"formula {f['formula_id']} missing unit")
    for fr in frels:
        if fr.get('formula_id') not in fids: errors.append(f"formula relation missing formula {fr.get('formula_id')}")
        if fr.get('chunk_id') not in cids: errors.append(f"formula relation missing chunk {fr.get('chunk_id')}")
    for cl in clusters:
        unknown=set(cl.get('member_exercise_ids',[]))-eids
        if unknown: errors.append(f"cluster {cl['related_cluster_id']} unknown exercises: {sorted(unknown)[:5]}")
    role_units=Counter(u['unit_type'] for u in units); role_chunks=Counter(c['unit_type'] for c in chunks);qcounts=Counter(x['flag'] for x in qf)
    # Source anomalies are expected but explicit.
    if qcounts.get('missing_end_enonceRAG'): warnings.append(f"source anomaly preserved: missing_end_enonceRAG={qcounts['missing_end_enonceRAG']}")
    if qcounts.get('unbalanced_double_dollar'): warnings.append(f"source anomaly preserved: unbalanced_double_dollar={qcounts['unbalanced_double_dollar']}")
    if unres: warnings.append(f'unresolved LaTeX refs: {len(unres)}')
    print('=== Exo7 v2 Structured RAG Validation ===')
    print(f'documents: {len(docs)} units: {len(units)} chunks: {len(chunks)}')
    print(f"statements: {role_units['exercise_statement']} hints: {role_units['exercise_hint']} solutions: {role_units['exercise_solution']}")
    print(f'formulas: {len(forms)} formal_objects: {len(formal)} relations: {len(rels)} topics: {len(topics)} related_clusters: {len(clusters)}')
    print(f'unresolved_refs: {len(unres)} max_chunk_chars: {max(len(c.get("latex","")) for c in chunks)}')
    print(f'quality_flags: {len(qf)} {dict(qcounts)}')
    print(f'errors: {len(errors)} warnings: {len(warnings)}')
    for x in errors[:20]:print('[ERROR]',x)
    for x in warnings:print('[WARN]',x)
    if errors: raise SystemExit(1)
    print('[OK] Exo7 v2 structured dataset valid')
if __name__=='__main__':main()
