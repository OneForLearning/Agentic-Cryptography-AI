from __future__ import annotations
import argparse, collections, hashlib, json, re, tempfile, uuid, zipfile
from pathlib import Path

PLACEHOLDER = "% >>> Insérer ici le texte, les formules, figures et listings de cette partie."

def sha(s:str)->str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sid(prefix:str,*parts,n=24)->str:
    return f"{prefix}-" + sha("\x1f".join(map(str,parts)))[:n]

def qid(s:str)->str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "ACE_AI_Crypto_Expert:"+s))

def write_jsonl(path:Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n")

def load_zip(zip_path:Path):
    td=tempfile.TemporaryDirectory()
    root=Path(td.name)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root)
    candidates=[p for p in root.rglob("main.tex")]
    if not candidates:
        raise RuntimeError("main.tex introuvable dans l'archive")
    project_root=candidates[0].parent
    return td, project_root

def main():
    ap=argparse.ArgumentParser(description="Audit et chunking navigation-only d'un squelette LaTeX Serious Cryptography.")
    ap.add_argument("zip_path", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args=ap.parse_args()

    td, root = load_zip(args.zip_path)
    try:
        tex_files=sorted(root.rglob("*.tex"))
        if not tex_files:
            raise RuntimeError("Aucun fichier .tex")

        documents=[]
        docmap={}
        for p in tex_files:
            rel=p.relative_to(root).as_posix()
            txt=p.read_text(encoding="utf-8")
            role="master" if rel=="main.tex" else ("frontmatter" if rel=="chapitres/preliminaires.tex" else "chapter")
            d={"document_id":sid("doc",rel),"source_file":rel,"document_role":role,
               "sha256":sha(txt),"source_chars":len(txt),"source_bytes":len(txt.encode("utf-8"))}
            documents.append(d); docmap[rel]=d

        main_path=root/"main.tex"
        main_text=main_path.read_text(encoding="utf-8")

        declared_envs=[]
        for line in main_text.splitlines():
            if "\\newtheorem" not in line:
                continue
            m=re.search(r'\\newtheorem(\*)?\{([^}]+)\}(?:\[([^\]]+)\])?\{([^}]+)\}(?:\[([^\]]+)\])?', line)
            if not m:
                continue
            env=m.group(2)
            usage=sum(len(re.findall(r'\\begin\{'+re.escape(env)+r'\}', p.read_text(encoding="utf-8"))) for p in tex_files)
            declared_envs.append({
                "environment":env,"display_name":m.group(4),"numbered":not bool(m.group(1)),
                "shared_counter":m.group(3),"within":m.group(5),"usage_count":usage,
            })

        macros=[]
        for line in main_text.splitlines():
            s=line.strip()
            m=re.match(r'\\newcommand\{\\([^}]+)\}(?:\[([0-9]+)\])?\{(.+?)\}(?:\s*%\s*(.*))?$',s)
            if m:
                macros.append({"name":"\\"+m.group(1),"arity":int(m.group(2) or 0),
                               "expansion":m.group(3),"comment":m.group(4),"kind":"macro"})
                continue
            m=re.match(r'\\DeclareMathOperator\{\\([^}]+)\}\{(.+?)\}(?:\s*%\s*(.*))?$',s)
            if m:
                macros.append({"name":"\\"+m.group(1),"arity":0,"expansion":m.group(2),
                               "comment":m.group(3),"kind":"math_operator"})

        units=[]; chunks=[]; relations=[]
        by_id={}
        for p in tex_files:
            rel=p.relative_to(root).as_posix()
            if rel=="main.tex":
                continue
            lines=p.read_text(encoding="utf-8").splitlines()
            stack={"chapter":None,"section":None,"subsection":None}
            file_units=[]
            for line_no,line in enumerate(lines,1):
                m=re.match(r'\s*\\(chapter|section|subsection)(\*)?\{(.+?)\}(?:\\label\{([^}]+)\})?',line)
                if not m:
                    continue
                typ,star,title,label=m.group(1),bool(m.group(2)),m.group(3),m.group(4)
                if typ=="chapter":
                    parent=None; stack["section"]=None; stack["subsection"]=None
                elif typ=="section":
                    parent=stack["chapter"]; stack["subsection"]=None
                else:
                    parent=stack["section"]
                if typ=="chapter":
                    path=[title]
                elif typ=="section":
                    ptitle=by_id[parent]["title"] if parent else None
                    path=[x for x in (ptitle,title) if x]
                else:
                    sec=by_id[parent]["title"] if parent else None
                    ch_id=by_id[parent]["parent_unit_id"] if parent else None
                    ch=by_id[ch_id]["title"] if ch_id else None
                    path=[x for x in (ch,sec,title) if x]
                uid=sid("outline",rel,line_no,typ,title)
                u={
                    "unit_id":uid,"document_id":docmap[rel]["document_id"],"source_file":rel,
                    "source_line_start":line_no,"unit_type":"outline_topic","heading_type":typ,
                    "title":title,"starred":star,"label":label,"parent_unit_id":parent,
                    "section_path":path,"content_kind":"outline","retrieval_role":"topic_router",
                    "knowledge_status":"placeholder_only","has_substantive_body":False,
                    "exclude_from_answer_generation":True
                }
                units.append(u); file_units.append(u); by_id[uid]=u
                if parent:
                    relations.append({
                        "relation_id":sid("rel","parent_of",parent,uid),"type":"parent_of",
                        "from_unit_id":parent,"to_unit_id":uid,"relation_strength":"structural",
                        "grounding_semantics":"outline_hierarchy_only"
                    })
                stack[typ]=uid

            groups=collections.defaultdict(list)
            for u in file_units:
                groups[(u["parent_unit_id"],u["heading_type"])].append(u)
            for group in groups.values():
                for a,b in zip(group,group[1:]):
                    relations.append({
                        "relation_id":sid("rel","next_sibling",a["unit_id"],b["unit_id"]),
                        "type":"next_sibling","from_unit_id":a["unit_id"],"to_unit_id":b["unit_id"],
                        "relation_strength":"structural","grounding_semantics":"document_order_only"
                    })

        children=collections.defaultdict(list)
        for r in relations:
            if r["type"]=="parent_of":
                children[r["from_unit_id"]].append(r["to_unit_id"])

        for u in units:
            u["child_unit_ids"]=children[u["unit_id"]]
            hierarchy=" > ".join(u["section_path"])
            emb=(
                "OUTLINE ONLY — no substantive source text is present for this heading.\n"
                f"Hierarchy: {hierarchy}\n"
                f"Topic heading: {u['title']}\n"
                "Use only for navigation or query routing. Do not use as evidence for a definition, "
                "formula, theorem, proof, proposition, lemma, corollary, property, hypothesis, or factual claim."
            )
            latex="\\"+u["heading_type"]+("*" if u["starred"] else "")+"{"+u["title"]+"}"
            cid=sid("chunk",u["unit_id"])
            chunks.append({
                "chunk_id":cid,"qdrant_id":qid(cid),"unit_id":u["unit_id"],"document_id":u["document_id"],
                "unit_type":"outline_topic","content_kind":"outline","retrieval_role":"topic_router",
                "knowledge_status":"placeholder_only","exclude_from_answer_generation":True,
                "source_file":u["source_file"],"source_line_start":u["source_line_start"],
                "section_path":u["section_path"],"title":u["title"],"heading_type":u["heading_type"],
                "parent_unit_id":u["parent_unit_id"],"related_unit_ids":u["child_unit_ids"],"label":u["label"],
                "latex":latex,"embedding_text":emb,"content_hash":sha(latex),"embedding_hash":sha(emb)
            })

        doc_relations=[]
        for m in re.finditer(r'\\input\{([^}]+)\}',main_text):
            target=m.group(1)+("" if m.group(1).endswith(".tex") else ".tex")
            if target in docmap:
                doc_relations.append({
                    "relation_id":sid("docrel","includes","main.tex",target),"type":"includes",
                    "from_document_id":docmap["main.tex"]["document_id"],
                    "to_document_id":docmap[target]["document_id"]
                })

        labels=[]
        for p in tex_files:
            rel=p.relative_to(root).as_posix()
            txt=p.read_text(encoding="utf-8")
            for m in re.finditer(r'\\label\{([^}]+)\}',txt):
                labels.append({"document_id":docmap[rel]["document_id"],"source_file":rel,"label":m.group(1)})

        content_files=[p for p in tex_files if p.name!="main.tex"]
        stats={
            "archive_tex_files":len(tex_files),"documents":len(documents),"outline_nodes":len(units),
            "outline_chunks":len(chunks),"knowledge_chunks":0,
            "chapters":sum(u["heading_type"]=="chapter" for u in units),
            "sections":sum(u["heading_type"]=="section" for u in units),
            "subsections":sum(u["heading_type"]=="subsection" for u in units),
            "placeholder_comments":sum(p.read_text(encoding="utf-8").count(PLACEHOLDER) for p in tex_files),
            "formal_environment_declarations":len(declared_envs),
            "formal_environment_usage":sum(e["usage_count"] for e in declared_envs),
            "math_macro_declarations":len(macros),
            "content_equation_environments":sum(len(re.findall(r'\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}',p.read_text(encoding="utf-8"))) for p in content_files),
            "content_inline_dollar_delimiters":sum(len(re.findall(r'(?<!\\)\$',p.read_text(encoding="utf-8"))) for p in content_files),
            "labels":len(labels),
            "refs":sum(len(re.findall(r'\\(?:ref|eqref|pageref)\{',p.read_text(encoding="utf-8"))) for p in tex_files),
            "document_includes":len(doc_relations),"structural_relations":len(relations),
        }

        args.out.mkdir(parents=True,exist_ok=True)
        write_jsonl(args.out/"documents.jsonl",documents)
        write_jsonl(args.out/"units.jsonl",units)
        write_jsonl(args.out/"relations.jsonl",relations)
        write_jsonl(args.out/"chunks.jsonl",chunks)
        write_jsonl(args.out/"declared_environments.jsonl",declared_envs)
        write_jsonl(args.out/"macros.jsonl",macros)
        write_jsonl(args.out/"document_relations.jsonl",doc_relations)
        write_jsonl(args.out/"labels.jsonl",labels)
        (args.out/"stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(stats,ensure_ascii=False,indent=2))
        print("[OK] Navigation-only outline dataset built.")
        print("[IMPORTANT] knowledge_chunks=0; do not merge these chunks into the factual/math answer corpus.")
    finally:
        td.cleanup()

if __name__=="__main__":
    main()
