from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description='Retrieve mathematical LaTeX context then ask a local Ollama LLM.')
    ap.add_argument('question')
    ap.add_argument('--llm',default='qwen3:14b')
    ap.add_argument('--retriever',type=Path,default=Path(__file__).with_name('retrieve_latex.py'))
    ap.add_argument('--processed',type=Path,default=Path('corpus/processed/latex'))
    ap.add_argument('--max-context-chars',type=int,default=30000)
    args=ap.parse_args()
    cmd=[sys.executable,str(args.retriever),args.question,'--processed',str(args.processed),'--max-context-chars',str(args.max_context_chars),'--json']
    raw=subprocess.check_output(cmd,text=True)
    chunks=json.loads(raw)
    context=[]
    for c in chunks:
        context.append(
            f"[CHUNK {c['chunk_id']}]\nCourse: {c.get('course_title')}\nSection: {c.get('section_path_text')}\nType: {c.get('unit_type')}\nLabel: {c.get('label')}\nSource file: {c.get('source_file')}\n\n{c.get('latex','')}"
        )
    joined='\n\n'.join(context)
    try:
        import ollama
    except ImportError as exc: raise SystemExit('uv add ollama') from exc
    system='''Tu réponds à partir du contexte RAG fourni. Le LaTeX source est l'autorité documentaire.\n\nRègles:\n- N'invente pas un résultat absent du contexte.\n- Pour un théorème, distingue l'énoncé de sa preuve.\n- Si une preuve est incomplète ou qu'un document est signalé incomplet, dis-le.\n- Conserve les hypothèses mathématiques.\n- Cite les identifiants [CHUNK ...] qui supportent les affirmations importantes.\n- Si plusieurs sources divergent, signale la divergence au lieu de les fusionner silencieusement.'''
    prompt=f"QUESTION:\n{args.question}\n\nCONTEXTE RAG:\n{joined}"
    resp=ollama.chat(model=args.llm,messages=[{'role':'system','content':system},{'role':'user','content':prompt}],options={'temperature':0.1})
    content=resp['message']['content'] if isinstance(resp,dict) else resp.message.content
    print(content)

if __name__=='__main__': main()
