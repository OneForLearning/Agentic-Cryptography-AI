
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

CORE=("documents.jsonl","units.jsonl","relations.jsonl","chunks.jsonl")

def read_jsonl(p):
    rows=[]
    with p.open(encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try: rows.append(json.loads(line))
                except Exception as e: raise RuntimeError(f"{p}:{i}: {e}")
    return rows
def write_jsonl(p,rows):
    with p.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
def canon(o): return json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def rid(payload): return "rel-"+hashlib.sha256(canon(payload).encode()).hexdigest()[:32]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",action="append",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"Sortie non vide: {a.out}")
    merged={k:[] for k in CORE}
    sidecars={}
    for src in a.input:
        print("[INPUT]",src)
        for name in CORE:
            merged[name].extend(read_jsonl(src/name))
        for p in src.glob("*.jsonl"):
            if p.name in CORE: continue
            sidecars.setdefault(p.name,[]).extend(read_jsonl(p))
    # strict uniqueness for foreign-key-bearing objects
    for name,key in [("documents.jsonl","document_id"),("units.jsonl","unit_id"),("chunks.jsonl","chunk_id")]:
        seen=set()
        for r in merged[name]:
            v=r.get(key)
            if not v or v in seen: raise RuntimeError(f"{name}: duplicate/missing {key}={v}")
            seen.add(v)
    # dedupe relations by payload excluding relation_id and regenerate deterministic global IDs
    rel_payloads={}
    for r in merged["relations.jsonl"]:
        payload={k:v for k,v in r.items() if k!="relation_id"}
        rel_payloads[canon(payload)]=payload
    rels=[]
    for payload in rel_payloads.values():
        x=dict(payload); x["relation_id"]=rid(payload); rels.append(x)
    merged["relations.jsonl"]=rels
    uid={u["unit_id"] for u in merged["units.jsonl"]}
    for r in rels:
        if r.get("from_unit_id") not in uid or r.get("to_unit_id") not in uid:
            raise RuntimeError(f"dangling relation {r}")
    a.out.mkdir(parents=True,exist_ok=True)
    for name in CORE: write_jsonl(a.out/name,merged[name])
    # sidecars: merge same-named records; they are advisory only
    for name,rows in sidecars.items(): write_jsonl(a.out/name,rows)
    stats={
        "inputs":[str(x) for x in a.input],
        "documents":len(merged["documents.jsonl"]),
        "units":len(merged["units.jsonl"]),
        "relations":len(merged["relations.jsonl"]),
        "chunks":len(merged["chunks.jsonl"]),
    }
    (a.out/"merge_stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    print("[OK]",a.out)
if __name__=="__main__": main()
