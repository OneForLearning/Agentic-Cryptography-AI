
from __future__ import annotations
import argparse, hashlib, json, re, uuid, zipfile
from collections import Counter, defaultdict
from pathlib import Path

FORMAL_ENVS = {
    "definition":"definition","remark":"remark","proposition":"proposition",
    "lemma":"lemma","theorem":"theorem","corollary":"corollary","example":"example",
    "theodef":"theorem_definition","theoreme":"theorem","propriete":"property",
    "corollaire":"corollary","lemme":"lemma","exemple":"example","exercice":"exercise",
}
GENERIC_TITLES = {
    "example","examples","exemple","exemples","exemple 1","exemple 2",
    "exemple 1 (bis)","exemple 2 (bis)","exemple (bis)","exemple (ter)"
}

def h(s:str,n=20): return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]
def uuid_from(s:str): return str(uuid.uuid5(uuid.NAMESPACE_URL, "ace-ai-crypto:"+s))
def dump(path, rows):
    with path.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
def strip_comments(s):
    return re.sub(r'(?m)(?<!\\)%.*$','',s)
def balanced_arg(s,start,op="{",cl="}"):
    if start>=len(s) or s[start]!=op: return None,None
    depth=0
    for i in range(start,len(s)):
        if s[i]==op: depth+=1
        elif s[i]==cl:
            depth-=1
            if depth==0: return s[start+1:i], i+1
    return None,None
def parse_heading(line):
    m=re.match(r'\s*\\(chapter|section|subsection|subsubsection)\*?',line)
    if not m: return None
    p=m.end()
    while p<len(line) and line[p].isspace(): p+=1
    if p>=len(line) or line[p]!='{': return None
    title,end=balanced_arg(line,p)
    if title is None: return None
    lm=re.search(r'\\label\{([^}]*)\}',line[end:])
    return m.group(1), title.strip(), (lm.group(1) if lm else None)
def parse_formal_line(line):
    m=re.search(r'\\begin\{([^}]+)\}',line)
    if not m or m.group(1) not in FORMAL_ENVS: return None
    env=m.group(1); rest=line[m.end():]
    title=None
    before_label=rest.split("\\label",1)[0]
    st=before_label.lstrip()
    if st.startswith("["):
        # Skeleton sources sometimes contain literal ] in notation such as F[x].
        # Use the final ] before \label/end-of-line as the intended boundary.
        a=before_label.find("[")
        b=before_label.rfind("]")
        if b>a: title=before_label[a+1:b].strip()
    elif env=="exemple" and st.startswith("{"):
        a=rest.find("{"); val,end=balanced_arg(rest,a)
        if val is not None: title=val.strip()
    lm=re.search(r'\\label\{([^}]*)\}',line)
    return env,title,(lm.group(1) if lm else None), ("optional_title_internal_bracket" if title and "]" in title else None)
def meaningful_title(title):
    if not title: return False
    norm=re.sub(r'\s+',' ',title).strip().lower()
    if norm in GENERIC_TITLES: return False
    if re.fullmatch(r'exemple(?:s)?(?:\s+\d+)?(?:\s*\([^)]*\))?',norm): return False
    return bool(re.search(r'[A-Za-zÀ-ÿ]', title))
def logical_doc(source):
    p=source.split("/")
    if len(p)<2: return "unknown"
    folder=p[1]
    return {
        "abstract-algebra-notes":"abstract_algebra_notes",
        "algebre_lineaire":"algebre_lineaire",
        "algebre_lineaire2":"algebre_lineaire2",
        "applications_lineaires":"applications_lineaires",
        "determinant_matrice":"determinant_matrice",
        "determinants":"determinants",
    }.get(folder, folder.replace("-","_"))
def title_for_doc(doc):
    return {
        "abstract_algebra_notes":"Abstract Algebra Notes (skeleton)",
        "algebre_lineaire":"Révisions d'algèbre linéaire (skeleton)",
        "algebre_lineaire2":"Compléments d'algèbre linéaire (skeleton)",
        "applications_lineaires":"Applications linéaires (skeleton)",
        "determinant_matrice":"Déterminant d'une matrice (skeleton)",
        "determinants":"Déterminants (skeleton)",
    }.get(doc,doc)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("archive",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    out=args.out; out.mkdir(parents=True,exist_ok=True)

    with zipfile.ZipFile(args.archive) as z:
        tex_names=[n for n in z.namelist() if n.lower().endswith(".tex")]
        texts={n:z.read(n).decode("utf-8",errors="replace") for n in tex_names}

    docs=[]
    for doc in sorted({logical_doc(n) for n in tex_names}):
        fs=[n for n in tex_names if logical_doc(n)==doc]
        docs.append({
            "document_id":"doc-"+h(doc),
            "title":title_for_doc(doc),
            "content_kind":"outline",
            "knowledge_status":"skeleton_only",
            "retrieval_role":"topic_router",
            "exclude_from_answer_generation":True,
            "source_files":fs,
        })
    docid={d["title"].split(" (skeleton)")[0]:d["document_id"] for d in docs}
    did_by_key={n: next(d["document_id"] for d in docs if n in d["source_files"]) for n in tex_names}

    # macros / declared envs
    macros=[]; envdecl=[]; includes=[]
    for n,data in texts.items():
        for m in re.finditer(r'\\(?:newcommand|renewcommand)\*?\{\\([A-Za-z@]+)\}(?:\[(\d+)\])?',data):
            macros.append({"source_file":n,"name":m.group(1),"arity":int(m.group(2) or 0)})
        for m in re.finditer(r'\\DeclareMathOperator\*?\{\\([A-Za-z@]+)\}\{([^}]*)\}',data):
            macros.append({"source_file":n,"name":m.group(1),"arity":0,"operator_text":m.group(2)})
        for pat,kind in [(r'\\newtheorem\{([^}]+)\}\{([^}]+)\}',"newtheorem"),
                         (r'\\defenv\{([^}]+)\}\{([^}]+)\}',"defenv"),
                         (r'\\boite\{([^}]+)\}\{([^}]+)\}',"boite")]:
            for m in re.finditer(pat,data):
                envdecl.append({"source_file":n,"environment":m.group(1),"display_name":m.group(2),"declaration_kind":kind})
        for m in re.finditer(r'\\input\{([^}]+)\}',data):
            includes.append({"source_file":n,"target":m.group(1)})

    units=[]; chunks=[]; relations=[]; placeholders=[]; quality=[]
    heading_count=Counter(); formal_count=Counter(); titled_count=Counter()
    # maintain sibling groups per doc/source hierarchy; abstract chapter files are sequential but hierarchy local per chapter.
    for n in tex_names:
        if n.endswith("/main.tex") and "abstract-algebra-notes" in n:
            continue
        data=texts[n]
        doc_key=logical_doc(n); document_id=did_by_key[n]
        lines=data.splitlines()
        offsets=[]; pos=0
        for line in lines:
            offsets.append(pos); pos += len(line)+1
        hierarchy={"chapter":None,"section":None,"subsection":None,"subsubsection":None}
        title_path=[]
        level_order={"chapter":1,"section":2,"subsection":3,"subsubsection":4}
        unit_at_level={}
        siblings=defaultdict(list)
        formal_by_parent=defaultdict(list)

        # body search helper
        full=data
        for i,line in enumerate(lines,1):
            if line.lstrip().startswith("%"): continue
            head=parse_heading(line)
            if head:
                typ,title,label=head
                lev=level_order[typ]
                # reset lower levels
                for k,v in level_order.items():
                    if v>=lev: unit_at_level.pop(k,None)
                parent=None
                for k,v in sorted(level_order.items(), key=lambda x:x[1], reverse=True):
                    if v<lev and k in unit_at_level:
                        parent=unit_at_level[k]; break
                uid="unit-"+h(f"{n}:{i}:{typ}:{title}:{label or ''}")
                path=[]
                if parent:
                    pu=next(u for u in units if u["unit_id"]==parent)
                    path=pu["section_path"]+[pu["title"]]
                unit={
                    "unit_id":uid,"document_id":document_id,"source_file":n,"source_line":i,
                    "unit_type":typ,"content_kind":"outline","retrieval_role":"topic_router",
                    "knowledge_status":"skeleton_only","exclude_from_answer_generation":True,
                    "title":title,"label":label,"section_path":path,
                }
                units.append(unit); heading_count[typ]+=1; unit_at_level[typ]=uid
                parentkey=parent or f"ROOT:{document_id}"
                siblings[parentkey].append(uid)
                if parent:
                    relations.append({"relation_id":"rel-"+h(f"parent:{parent}:{uid}"),"type":"parent_of",
                                      "from_unit_id":parent,"to_unit_id":uid,"strength":"structural"})
                emb="\n".join([
                    "Navigation-only mathematical outline.",
                    f"Document: {title_for_doc(doc_key)}",
                    "Path: "+(" > ".join(path+[title])),
                    f"Node type: {typ}",
                    "WARNING: source is a skeleton; this title is not a mathematical statement."
                ])
                cid="chunk-"+h(uid+":outline")
                chunks.append({
                    "chunk_id":cid,"unit_id":uid,"document_id":document_id,"unit_type":typ,
                    "content_kind":"outline","retrieval_role":"topic_router","knowledge_status":"skeleton_only",
                    "exclude_from_answer_generation":True,"source_file":n,"source_line_start":i,
                    "source_line_end":i,"section_path":path,"title":title,"label":label,
                    "embedding_text":emb,"latex":line.strip(),"content_hash":hashlib.sha256(line.strip().encode()).hexdigest(),
                    "qdrant_id":uuid_from(cid),
                })
                continue

            form=parse_formal_line(line)
            if form:
                env,title,label,qflag=form
                sem=FORMAL_ENVS[env]
                # parent deepest current heading
                parent=None
                for k,v in sorted(level_order.items(), key=lambda x:x[1], reverse=True):
                    if k in unit_at_level: parent=unit_at_level[k]; break
                uid="unit-"+h(f"{n}:{i}:{env}:{title or ''}:{label or ''}")
                # determine body emptiness conservatively
                start_off=offsets[i-1]+line.find("\\begin")
                end_token=f"\\end{{{env}}}"
                endpos=full.find(end_token,start_off)
                body=""
                if endpos!=-1:
                    begin_line_end=full.find("\n",start_off)
                    if begin_line_end==-1: begin_line_end=start_off+len(line)
                    body=full[begin_line_end+1:endpos]
                clean=strip_comments(body)
                clean=re.sub(r'\\label\{[^}]*\}','',clean)
                clean=re.sub(r'\s+',' ',clean).strip()
                is_empty=(clean=="")
                path=[]
                if parent:
                    pu=next(u for u in units if u["unit_id"]==parent)
                    path=pu["section_path"]+[pu["title"]]
                unit={
                    "unit_id":uid,"document_id":document_id,"source_file":n,"source_line":i,
                    "unit_type":"formal_placeholder","semantic_role":sem,"source_environment":env,
                    "content_kind":"outline","retrieval_role":"topic_router","knowledge_status":"placeholder_only",
                    "exclude_from_answer_generation":True,"title":title,"label":label,"section_path":path,
                    "body_empty":is_empty,
                }
                units.append(unit); formal_count[sem]+=1
                placeholders.append(dict(unit))
                if parent:
                    relations.append({"relation_id":"rel-"+h(f"parent:{parent}:{uid}"),"type":"parent_of",
                                      "from_unit_id":parent,"to_unit_id":uid,"strength":"structural"})
                    formal_by_parent[parent].append(uid)
                if qflag:
                    quality.append({"flag_id":"flag-"+h(uid+qflag),"unit_id":uid,"type":qflag,
                                    "source_file":n,"source_line":i,
                                    "message":"Optional argument contains a literal ] before the final delimiter; preserve as routing metadata only."})
                if clean:
                    quality.append({"flag_id":"flag-"+h(uid+"nonempty"),"unit_id":uid,"type":"unexpected_nonempty_formal_body",
                                    "source_file":n,"source_line":i,"message":"Formal body appears non-empty; manual inspection required."})
                if meaningful_title(title):
                    titled_count[sem]+=1
                    emb="\n".join([
                        "Navigation-only mathematical formal-object title.",
                        f"Document: {title_for_doc(doc_key)}",
                        "Path: "+(" > ".join(path) if path else "(document root)"),
                        f"Formal type: {sem}",
                        f"Title: {title}",
                        f"Label: {label or '(none)'}",
                        "WARNING: the source body is empty. Do not use this chunk as an equation, definition, theorem statement, proof, or factual source."
                    ])
                    cid="chunk-"+h(uid+":formal-title")
                    chunks.append({
                        "chunk_id":cid,"unit_id":uid,"document_id":document_id,"unit_type":"formal_placeholder",
                        "semantic_role":sem,"content_kind":"outline","retrieval_role":"topic_router",
                        "knowledge_status":"title_only","exclude_from_answer_generation":True,
                        "source_file":n,"source_line_start":i,"source_line_end":i,
                        "section_path":path,"title":title,"label":label,
                        "embedding_text":emb,"latex":line.strip(),
                        "content_hash":hashlib.sha256(line.strip().encode()).hexdigest(),"qdrant_id":uuid_from(cid)
                    })

        for group, ids in siblings.items():
            for a,b in zip(ids,ids[1:]):
                relations.append({"relation_id":"rel-"+h(f"next:{a}:{b}"),"type":"next_sibling",
                                  "from_unit_id":a,"to_unit_id":b,"strength":"structural_weak"})
        for parent, ids in formal_by_parent.items():
            for a,b in zip(ids,ids[1:]):
                relations.append({"relation_id":"rel-"+h(f"nextformal:{a}:{b}"),"type":"next_formal_placeholder",
                                  "from_unit_id":a,"to_unit_id":b,"strength":"structural_weak"})

    # resolve unexpected bodies: known parser artifact due optional args with ] may have appeared.
    # Since source explicitly declares outline_only, any content remaining inside env should not be knowledge.
    # Count actual equations/references after comments.
    formulas=0; refs=0
    for n,data in texts.items():
        nc=strip_comments(data)
        formulas += len(re.findall(r'\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}',nc))
        formulas += len(re.findall(r'\\\[',nc)) + len(re.findall(r'\$\$',nc))
        refs += len(re.findall(r'\\(?:ref|eqref|pageref)\{',nc))

    # unique ids checks
    for rows,key in [(units,"unit_id"),(chunks,"chunk_id"),(relations,"relation_id")]:
        vals=[r[key] for r in rows]
        if len(vals)!=len(set(vals)): raise RuntimeError(f"duplicate {key}")

    stats={
        "archive_tex_files":len(tex_names),"logical_documents":len(docs),
        "outline_heading_units":sum(heading_count.values()),
        "formal_placeholders":sum(formal_count.values()),
        "titled_formal_router_chunks":sum(titled_count.values()),
        "chunks":len(chunks),"knowledge_chunks":sum(1 for c in chunks if not c["exclude_from_answer_generation"]),
        "heading_types":dict(heading_count),"formal_types":dict(formal_count),
        "titled_formal_types":dict(titled_count),"relations":len(relations),
        "relation_types":dict(Counter(r["type"] for r in relations)),
        "macro_declarations":len(macros),"declared_environments":len(envdecl),
        "content_display_math_blocks":formulas,"content_refs":refs,
        "quality_flags":len(quality),"quality_flag_types":dict(Counter(q["type"] for q in quality)),
        "indexing_policy":"outline_only",
    }
    dump(out/"documents.jsonl",docs); dump(out/"units.jsonl",units); dump(out/"relations.jsonl",relations)
    dump(out/"chunks.jsonl",chunks); dump(out/"macros.jsonl",macros); dump(out/"declared_environments.jsonl",envdecl)
    dump(out/"formal_placeholders.jsonl",placeholders); dump(out/"quality_flags.jsonl",quality); dump(out/"includes.jsonl",includes)
    (out/"stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
