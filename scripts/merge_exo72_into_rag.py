from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

CORE=('documents.jsonl','macros.jsonl','units.jsonl','relations.jsonl','chunks.jsonl')
SIDECARS=('formulas.jsonl','formula_relations.jsonl','formal_objects.jsonl','topics.jsonl','exercise_topics.jsonl','semantic_annotations.jsonl','related_clusters.jsonl','duplicate_groups.jsonl','quality_flags.jsonl','unresolved_refs.jsonl','stats.json')

def readj(path):
 out=[]
 with path.open(encoding='utf-8') as f:
  for line in f:
   if line.strip(): out.append(json.loads(line))
 return out

def writej(path,records):
 with path.open('w',encoding='utf-8') as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')

def unique(records,key,label):
 seen=set()
 for r in records:
  v=r.get(key)
  if not v: raise RuntimeError(f'{label}: missing {key}')
  if v in seen: raise RuntimeError(f'{label}: duplicate {key}={v}')
  seen.add(v)

def normalize_relations(records):
 seen=set();out=[]
 for r in records:
  payload={k:v for k,v in r.items() if k!='relation_id'}
  s=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))
  if s in seen: continue
  seen.add(s)
  nr=dict(payload);nr['relation_id']='rel-'+hashlib.sha256(s.encode()).hexdigest()[:32]
  out.append(nr)
 unique(out,'relation_id','relations')
 return out

def merge_macros(records):
 # Exact duplicates are removed. Same macro_id with differing payload is safely re-namespaced.
 seen_payload=set();seen_id=set();out=[]
 for r in records:
  payload=json.dumps({k:v for k,v in r.items() if k!='macro_id'},ensure_ascii=False,sort_keys=True,separators=(',',':'))
  if payload in seen_payload: continue
  seen_payload.add(payload)
  nr=dict(r);mid=nr.get('macro_id') or 'macro-'+hashlib.sha256(payload.encode()).hexdigest()[:24]
  if mid in seen_id: mid='macro-'+hashlib.sha256((mid+'|'+payload).encode()).hexdigest()[:24]
  nr['macro_id']=mid;seen_id.add(mid);out.append(nr)
 return out

def main():
 ap=argparse.ArgumentParser(description='Merge a validated base RAG dataset with Exo7 v2 while preserving Exo7 sidecars under exo72_* names.')
 ap.add_argument('--base',type=Path,required=True);ap.add_argument('--exo72',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 for root in (args.base,args.exo72):
  for name in CORE:
   if not (root/name).exists(): raise SystemExit(f'[ERROR] missing {root/name}')
 if args.out.exists():
  if any(args.out.iterdir()): raise SystemExit(f'[ERROR] output is non-empty: {args.out}')
 else: args.out.mkdir(parents=True)
 # preserve every non-core file from base first
 for p in args.base.iterdir():
  if p.is_file() and p.name not in CORE:
   shutil.copy2(p,args.out/p.name)
 base={n:readj(args.base/n) for n in CORE};exo={n:readj(args.exo72/n) for n in CORE}
 docs=base['documents.jsonl']+exo['documents.jsonl'];units=base['units.jsonl']+exo['units.jsonl'];chunks=base['chunks.jsonl']+exo['chunks.jsonl'];macros=merge_macros(base['macros.jsonl']+exo['macros.jsonl']);rels=normalize_relations(base['relations.jsonl']+exo['relations.jsonl'])
 unique(docs,'document_id','documents');unique(units,'unit_id','units');unique(chunks,'chunk_id','chunks')
 uids={u['unit_id'] for u in units};dids={d['document_id'] for d in docs}
 for u in units:
  if u.get('document_id') not in dids: raise RuntimeError(f"orphan unit {u['unit_id']}")
 for c in chunks:
  if c.get('unit_id') not in uids: raise RuntimeError(f"orphan chunk {c['chunk_id']}")
 for r in rels:
  if r.get('from_unit_id') and r['from_unit_id'] not in uids: raise RuntimeError(f"orphan relation from {r['relation_id']}")
  if r.get('resolved') and r.get('to_unit_id') and r['to_unit_id'] not in uids: raise RuntimeError(f"orphan relation to {r['relation_id']}")
 writej(args.out/'documents.jsonl',docs);writej(args.out/'macros.jsonl',macros);writej(args.out/'units.jsonl',units);writej(args.out/'relations.jsonl',rels);writej(args.out/'chunks.jsonl',chunks)
 # Prefix Exo7 sidecars to avoid collisions with prior math/corpus sidecars.
 for name in SIDECARS:
  src=args.exo72/name
  if src.exists(): shutil.copy2(src,args.out/('exo72_'+name))
 stats={'base':str(args.base),'exo72':str(args.exo72),'documents':len(docs),'units':len(units),'relations':len(rels),'chunks':len(chunks),'added_exo72_chunks':len(exo['chunks.jsonl'])}
 (args.out/'merge_exo72_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(stats,ensure_ascii=False,indent=2));print('[OK] merged dataset:',args.out)
if __name__=='__main__':main()
