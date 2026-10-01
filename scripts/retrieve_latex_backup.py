from __future__ import annotations
import argparse,json,re,sqlite3
from collections import defaultdict
from pathlib import Path


def response_embeddings(resp):
    if isinstance(resp,dict): return resp.get('embeddings') or resp.get('embedding')
    if hasattr(resp,'embeddings'): return resp.embeddings
    if hasattr(resp,'embedding'): return [resp.embedding]
    raise RuntimeError('Unknown Ollama embedding response')

def rrf(rankings,k=60):
    score=defaultdict(float)
    for ranking in rankings:
        for rank,cid in enumerate(ranking,1): score[cid]+=1.0/(k+rank)
    return score

def fts_query(q):
    terms=re.findall(r'[\wÀ-ÿ]+',q,flags=re.UNICODE)
    return ' OR '.join('"'+t.replace('"','')+'"' for t in terms if len(t)>1) or '"'+q.replace('"','')+'"'

def qdrant_hits(client,collection,vector,limit):
    try:
        res=client.query_points(collection_name=collection,query=vector,limit=limit,with_payload=True)
        pts=getattr(res,'points',res)
    except Exception:
        pts=client.search(collection_name=collection,query_vector=vector,limit=limit,with_payload=True)
    out=[]
    for p in pts:
        payload=getattr(p,'payload',{}) or {}
        cid=payload.get('chunk_id')
        if cid: out.append((cid,float(getattr(p,'score',0.0)),payload))
    return out

def type_bonus(query,unit_type):
    q=query.lower()
    if any(w in q for w in ['démontr','preuve','prove','proof']) and unit_type=='proof': return 0.12
    if any(w in q for w in ['défin','definition','what is','qu’est-ce']) and unit_type=='definition': return 0.12
    if any(w in q for w in ['théorème','theorem']) and unit_type=='theorem': return 0.08
    if any(w in q for w in ['lemme','lemma']) and unit_type=='lemma': return 0.08
    if any(w in q for w in ['corollaire','corollary']) and unit_type=='corollary': return 0.08
    return 0.0

def main():
    ap=argparse.ArgumentParser(description='Hybrid dense+FTS retrieval with theorem/proof graph expansion.')
    ap.add_argument('query')
    ap.add_argument('--processed',type=Path,default=Path('corpus/processed/latex'))
    ap.add_argument('--model',default='bge-m3')
    ap.add_argument('--ollama-host',default='http://127.0.0.1:11434')
    ap.add_argument('--qdrant-url',default='http://127.0.0.1:6333')
    ap.add_argument('--collection',default='livre_crypto_latex_v1')
    ap.add_argument('--dense-k',type=int,default=24)
    ap.add_argument('--lexical-k',type=int,default=24)
    ap.add_argument('--top-k',type=int,default=12)
    ap.add_argument('--max-context-chars',type=int,default=30000)
    ap.add_argument('--json',action='store_true')
    args=ap.parse_args()
    try:
        import ollama
        from qdrant_client import QdrantClient
    except ImportError as exc: raise SystemExit('uv add ollama qdrant-client') from exc
    oclient=ollama.Client(host=args.ollama_host)
    qclient=QdrantClient(url=args.qdrant_url)
    vector=response_embeddings(oclient.embed(model=args.model,input=[args.query]))[0]
    dense=qdrant_hits(qclient,args.collection,vector,args.dense_k)
    con=sqlite3.connect(args.processed/'lexical.db');con.row_factory=sqlite3.Row
    try:
        try:
            rows=con.execute('SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?', (fts_query(args.query),args.lexical_k)).fetchall()
        except sqlite3.OperationalError:
            rows=[]
        lexical=[r['chunk_id'] for r in rows]
        dense_ids=[x[0] for x in dense]
        scores=rrf([dense_ids,lexical])
        dense_payload={cid:p for cid,_,p in dense}
        # Retrieve payloads for all candidates from SQLite; more authoritative than Qdrant payload.
        candidate_ids=set(scores)
        payloads={}
        for cid in candidate_ids:
            row=con.execute('SELECT payload_json FROM chunks WHERE chunk_id=?',(cid,)).fetchone()
            if row: payloads[cid]=json.loads(row[0])
        for cid,p in payloads.items(): scores[cid]+=type_bonus(args.query,p.get('unit_type'))
        ranked=sorted(scores,key=scores.get,reverse=True)
        selected=[];doc_counts=defaultdict(int);seen_hash=set();chars=0
        def add_chunk(cid,reason):
            nonlocal chars
            if any(x['chunk_id']==cid for x in selected): return
            row=con.execute('SELECT payload_json FROM chunks WHERE chunk_id=?',(cid,)).fetchone()
            if not row: return
            p=json.loads(row[0]);h=p.get('content_hash')
            if h in seen_hash: return
            if doc_counts[p['document_id']]>=4 and reason=='base': return
            cost=len(p.get('latex',''))
            if chars+cost>args.max_context_chars and selected: return
            p['_retrieval_reason']=reason;p['_fusion_score']=scores.get(cid,0.0)
            selected.append(p);seen_hash.add(h);doc_counts[p['document_id']]+=1;chars+=cost
        for cid in ranked[:args.top_k]:
            add_chunk(cid,'base')
            if not selected: continue
            # Expand proof<->statement and explicit references.
            p=next((x for x in selected if x['chunk_id']==cid),None)
            if not p: continue
            unit=p['unit_id']
            related_units=set(p.get('related_unit_ids',[]))
            if p.get('proof_of'): related_units.add(p['proof_of'])
            relrows=con.execute("SELECT to_unit_id FROM relations WHERE from_unit_id=? AND type IN ('proof_of','references') AND to_unit_id IS NOT NULL",(unit,)).fetchall()
            related_units.update(r[0] for r in relrows)
            back=con.execute("SELECT from_unit_id FROM relations WHERE to_unit_id=? AND type='proof_of'",(unit,)).fetchall()
            related_units.update(r[0] for r in back)
            for uid in list(related_units)[:4]:
                crow=con.execute('SELECT chunk_id FROM chunks WHERE unit_id=? ORDER BY chunk_id LIMIT 2',(uid,)).fetchall()
                for r in crow: add_chunk(r[0],'structural')
        if args.json:
            print(json.dumps(selected,indent=2,ensure_ascii=False))
        else:
            print(f'Query: {args.query}\nResults: {len(selected)}\n')
            for i,p in enumerate(selected,1):
                print('='*88)
                print(f'[{i}] {p["chunk_id"]} score={p.get("_fusion_score",0):.5f} reason={p["_retrieval_reason"]}')
                print('Course:',p.get('course_title'))
                print('Section:',p.get('section_path_text'))
                print('Type:',p.get('unit_type'),'Label:',p.get('label'),'Proof of:',p.get('proof_of'))
                print('-'*88)
                print(p.get('latex','')[:5000])
    finally: con.close()

if __name__=='__main__': main()
