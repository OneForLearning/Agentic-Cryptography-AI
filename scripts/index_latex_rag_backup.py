from __future__ import annotations
import argparse, json, sqlite3, time
from pathlib import Path
from typing import Iterable

from latex_rag_core import read_jsonl


def batched(seq, n):
    for i in range(0,len(seq),n): yield seq[i:i+n]

def response_embeddings(resp):
    if isinstance(resp,dict): return resp.get('embeddings') or resp.get('embedding')
    if hasattr(resp,'embeddings'): return resp.embeddings
    if hasattr(resp,'embedding'): return [resp.embedding]
    raise RuntimeError('Unknown Ollama embedding response')

def build_sqlite(db_path:Path,chunks:list[dict],relations:list[dict]):
    db_path.parent.mkdir(parents=True,exist_ok=True)
    if db_path.exists(): db_path.unlink()
    con=sqlite3.connect(db_path)
    try:
        con.execute('PRAGMA journal_mode=WAL')
        con.execute('CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, unit_id TEXT, document_id TEXT, unit_type TEXT, payload_json TEXT NOT NULL, embedding_text TEXT NOT NULL)')
        con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, document_id UNINDEXED, unit_type UNINDEXED, embedding_text, tokenize='unicode61')")
        con.execute('CREATE TABLE relations (type TEXT, from_unit_id TEXT, to_unit_id TEXT, target_label TEXT, resolved INTEGER)')
        for c in chunks:
            payload=json.dumps(c,ensure_ascii=False,separators=(',',':'))
            con.execute('INSERT INTO chunks VALUES (?,?,?,?,?,?)',(c['chunk_id'],c['unit_id'],c['document_id'],c['unit_type'],payload,c['embedding_text']))
            con.execute('INSERT INTO chunks_fts VALUES (?,?,?,?)',(c['chunk_id'],c['document_id'],c['unit_type'],c['embedding_text']))
        for r in relations:
            con.execute('INSERT INTO relations VALUES (?,?,?,?,?)',(r.get('type'),r.get('from_unit_id'),r.get('to_unit_id'),r.get('target_label'),1 if r.get('resolved',True) else 0))
        con.execute('CREATE INDEX idx_chunks_unit ON chunks(unit_id)')
        con.execute('CREATE INDEX idx_rel_from ON relations(from_unit_id)')
        con.execute('CREATE INDEX idx_rel_to ON relations(to_unit_id)')
        con.commit()
    finally: con.close()

def main():
    ap=argparse.ArgumentParser(description='Embed LaTeX chunks with Ollama bge-m3 and index in Qdrant + SQLite FTS5.')
    ap.add_argument('--processed',type=Path,default=Path('corpus/processed/latex'))
    ap.add_argument('--model',default='bge-m3')
    ap.add_argument('--ollama-host',default='http://127.0.0.1:11434')
    ap.add_argument('--qdrant-url',default='http://127.0.0.1:6333')
    ap.add_argument('--collection',default='livre_crypto_latex_v1')
    ap.add_argument('--batch-size',type=int,default=24)
    ap.add_argument('--recreate',action='store_true')
    args=ap.parse_args()
    try:
        import ollama
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise SystemExit('Install dependencies: uv add ollama qdrant-client') from exc
    chunks=read_jsonl(args.processed/'chunks.jsonl')
    relations=read_jsonl(args.processed/'relations.jsonl')
    if not chunks: raise SystemExit('No chunks')
    build_sqlite(args.processed/'lexical.db',chunks,relations)
    print(f'[OK] SQLite FTS5: {(args.processed/"lexical.db").resolve()}')
    oclient=ollama.Client(host=args.ollama_host)
    first_resp=oclient.embed(model=args.model,input=[chunks[0]['embedding_text']])
    first_vec=response_embeddings(first_resp)[0]
    dim=len(first_vec)
    print('Embedding dimension:',dim)
    qclient=QdrantClient(url=args.qdrant_url)
    exists=qclient.collection_exists(args.collection)
    if exists and args.recreate:
        qclient.delete_collection(args.collection); exists=False
    if not exists:
        qclient.create_collection(collection_name=args.collection,vectors_config=models.VectorParams(size=dim,distance=models.Distance.COSINE))
    # best-effort payload indexes
    for field in ['document_id','unit_type','language','source_file','label','content_hash']:
        try: qclient.create_payload_index(args.collection,field_name=field,field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception: pass
    total=len(chunks); done=0
    for batch in batched(chunks,args.batch_size):
        texts=[c['embedding_text'] for c in batch]
        resp=oclient.embed(model=args.model,input=texts)
        vectors=response_embeddings(resp)
        if len(vectors)!=len(batch): raise RuntimeError('Embedding batch size mismatch')
        points=[]
        for c,v in zip(batch,vectors):
            payload={k:v2 for k,v2 in c.items() if k not in {'embedding_text','latex'}}
            # Keep text in Qdrant too; retrieval can work even without SQLite payload lookup.
            payload['embedding_text']=c['embedding_text'];payload['latex']=c['latex']
            points.append(models.PointStruct(id=c['qdrant_id'],vector=v,payload=payload))
        qclient.upsert(collection_name=args.collection,points=points,wait=True)
        done+=len(batch); print(f'Indexed {done}/{total}')
    print(f'[OK] Qdrant collection={args.collection} points={total} model={args.model}')

if __name__=='__main__': main()
