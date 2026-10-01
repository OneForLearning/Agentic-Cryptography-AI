from __future__ import annotations
import argparse, bisect, hashlib, json, re, uuid
from collections import Counter, defaultdict
from pathlib import Path

def sha256_text(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def stable_id(prefix,*parts): return prefix+'-'+hashlib.sha256('\x1f'.join(parts).encode()).hexdigest()[:20]
def qdrant_uuid(v): return str(uuid.uuid5(uuid.NAMESPACE_URL,'ace-ai-crypto:'+v))
LINE_POSITIONS=[]
def line_number(text,pos): return bisect.bisect_right(LINE_POSITIONS,max(0,pos))+1

def read_braced(text,pos):
    if pos>=len(text) or text[pos]!='{': return None
    d=0;i=pos;start=pos+1
    while i<len(text):
        if text[i]=='\\': i+=2; continue
        if text[i]=='{': d+=1
        elif text[i]=='}':
            d-=1
            if d==0: return text[start:i],i+1
        i+=1
    return None

def skip_ws(text,pos):
    while pos<len(text) and text[pos].isspace(): pos+=1
    return pos

def mask_comments(text):
    out=list(text)
    for m in re.finditer(r'(?m)(?<!\\)%[^\n]*',text):
        out[m.start():m.end()]=' '*(m.end()-m.start())
    return ''.join(out)

def repair_embedding_text(s):
    repl={'Ã©':'é','Ã¨':'è','Ãª':'ê','Ã«':'ë','Ã‰':'É','Ã€':'À','Ã¢':'â','Ã®':'î','Ã¯':'ï','Ã´':'ô','Ã¶':'ö','Ã¹':'ù','Ã»':'û','Ã¼':'ü','Ã§':'ç','Å“':'œ','â€™':'’','â€“':'–','â€”':'—','Â°':'°','Â«':'«','Â»':'»','Â ':' '}
    out=s;n=0
    for a,b in repl.items():
        c=out.count(a)
        if c: out=out.replace(a,b);n+=c
    return out,n

def parse_newcommands(text,source_file,doc_id):
    out=[];masked=mask_comments(text)
    for m in re.finditer(r'\\(newcommand|renewcommand|providecommand)\*?',masked):
        pos=skip_ws(masked,m.end());name=None
        if pos<len(masked) and masked[pos]=='{':
            br=read_braced(masked,pos)
            if not br: continue
            rawname,pos=br; mm=re.search(r'\\([A-Za-z@]+)',rawname);name=mm.group(1) if mm else None
        elif pos<len(masked) and masked[pos]=='\\':
            mm=re.match(r'\\([A-Za-z@]+)',masked[pos:])
            if mm: name=mm.group(1);pos+=len(mm.group(0))
        if not name: continue
        pos=skip_ws(masked,pos);nargs=0;default=None
        if pos<len(masked) and masked[pos]=='[':
            e=masked.find(']',pos)
            if e!=-1:
                v=masked[pos+1:e].strip();pos=e+1
                if v.isdigit(): nargs=int(v)
        pos=skip_ws(masked,pos)
        if pos<len(masked) and masked[pos]=='[':
            e=masked.find(']',pos)
            if e!=-1: default=text[pos+1:e];pos=e+1
        pos=skip_ws(masked,pos);br=read_braced(masked,pos)
        if not br: continue
        _,end=br
        out.append({'command':name,'kind':m.group(1),'num_args':nargs,'default_arg':default,'definition':text[pos+1:end-1],'document_id':doc_id,'source_file':source_file,'line':line_number(text,m.start())})
    return out

def extract_commands(s): return set(re.findall(r'\\([A-Za-z@]+)',s))

def parse_braced_commands(masked,raw,cmd):
    out=[]
    for m in re.finditer(r'\\'+re.escape(cmd)+r'\{',masked):
        br=read_braced(masked,m.end()-1)
        if br: out.append((m.start(),raw[m.end():br[1]-1].strip(),br[1]))
    return out

def parse_role_blocks(raw,masked,role):
    if role=='statement': startcmd='enonce'; end_re=re.compile(r'\\finenonce\{([^}]+)\}')
    elif role=='hint': startcmd='indication'; end_re=re.compile(r'\\finindication\b')
    else: startcmd='correction'; end_re=re.compile(r'\\fincorrection\b')
    starts=[]
    for m in re.finditer(r'\\'+startcmd+r'\{',masked):
        br=read_braced(masked,m.end()-1)
        if not br: continue
        eid,pos=br;eid=eid.strip();title=None
        if role=='statement':
            pos=skip_ws(masked,pos);br2=read_braced(masked,pos)
            if not br2: continue
            title=raw[pos+1:br2[1]-1].strip();pos=br2[1]
        starts.append((m.start(),eid,pos,title))
    ends=[(m.start(),m.end(),m.group(1).strip() if role=='statement' else None) for m in end_re.finditer(masked)]
    end_starts=[x[0] for x in ends]
    out=[]
    for i,(st,eid,body_start,title) in enumerate(starts):
        next_start=starts[i+1][0] if i+1<len(starts) else len(raw)
        j=bisect.bisect_left(end_starts,body_start)
        anomaly=None
        if j<len(ends) and ends[j][0]<next_start and (role!='statement' or ends[j][2]==eid):
            body_end=ends[j][0];finish=ends[j][1]
        else:
            # for malformed statement, stop before next \exercice if available
            stop=next_start
            if role=='statement':
                k=masked.rfind('\\exercice{',body_start,next_start)
                if k!=-1: stop=k
            body_end=stop;finish=stop;anomaly='missing_'+('finenonce' if role=='statement' else ('finindication' if role=='hint' else 'fincorrection'))
        out.append({'exercise_id':eid,'role':role,'title':title,'body':raw[body_start:body_end].strip(),'start':st,'end':finish,'anomaly':anomaly})
    return out

def split_section_title(s):
    if not s: return None,None
    m=re.match(r'\s*([0-9]+(?:\.[0-9]+)?)\s+(.*)',s)
    return (m.group(1),m.group(2).strip()) if m else (None,s)

def top_level_item_parts(s):
    token=re.compile(r'\\begin\{([^}]+)\}|\\end\{([^}]+)\}|\\item\b')
    stack=[];positions=[];outer=None
    for m in token.finditer(s):
        if m.group(1):
            stack.append(m.group(1))
            if m.group(1)=='enumerate' and outer is None: outer=len(stack)
        elif m.group(2):
            env=m.group(2)
            if env in stack:
                while stack:
                    x=stack.pop()
                    if x==env: break
            if env=='enumerate' and outer and len(stack)<outer: outer=None
        elif outer is not None and len(stack)==outer and stack[-1]=='enumerate': positions.append(m.start())
    if len(positions)<2: return []
    prefix=s[:positions[0]].strip();parts=[]
    for i,p in enumerate(positions):
        q=positions[i+1] if i+1<len(positions) else len(s);piece=s[p:q].strip()
        if i==0 and prefix: piece=prefix+'\n'+piece
        parts.append(piece)
    return parts

def paragraphs(s): return [p.strip() for p in re.split(r'\n\s*\n+',s) if p.strip()]

def split_role(s,target,max_chars):
    if len(s)<=max_chars: return [s],False
    atoms=top_level_item_parts(s) or paragraphs(s);expanded=[];forced=False
    for a in atoms:
        if len(a)<=max_chars: expanded.append(a);continue
        ps=paragraphs(a)
        if len(ps)>1 and max(map(len,ps))<len(a): expanded.extend(ps);continue
        rest=a
        while len(rest)>max_chars:
            cut=rest.rfind('\n',0,max_chars)
            if cut<max_chars//2: cut=rest.rfind(' ',0,max_chars)
            if cut<max_chars//2: cut=max_chars
            expanded.append(rest[:cut].strip());rest=rest[cut:].strip();forced=True
        if rest: expanded.append(rest)
    chunks=[];cur=''
    for a in expanded:
        cand=(cur+'\n\n'+a).strip() if cur else a
        if cur and len(cand)>target: chunks.append(cur);cur=a
        else: cur=cand
    if cur: chunks.append(cur)
    return chunks,forced

def main():
    ap=argparse.ArgumentParser(description='Build exercise-aware RAG dataset from Exo7 sanitized LaTeX.')
    ap.add_argument('source',type=Path);ap.add_argument('--out',type=Path,default=Path('corpus/processed/exo7_v1'))
    ap.add_argument('--source-file',default='exo7_exercises/exercciceexo7.tex');ap.add_argument('--target-chars',type=int,default=4500);ap.add_argument('--max-chars',type=int,default=6000)
    args=ap.parse_args();raw=args.source.read_text(encoding='utf-8',errors='strict')
    global LINE_POSITIONS
    LINE_POSITIONS=[m.start() for m in re.finditer('\n',raw)]
    masked=mask_comments(raw)
    sha=sha256_text(raw);doc_id=stable_id('doc','Exo7',sha);source_file=args.source_file
    doc={'document_id':doc_id,'title':'Exo7 - Exercices de mathématiques','author':'Exo7','language':'fr','documentclass':'article','source_group':'exo7_corrected_exercises','root_file':source_file,'component_files':[source_file],'root_sha256':sha,'complete_inputs':True,'missing_inputs':[],'macro_context_complete':True,'content_kind':'exercise_corpus'}
    macros=parse_newcommands(raw,source_file,doc_id);macro_map={m['command']:m for m in macros}
    sections=parse_braced_commands(masked,raw,'section');secstarts=[x[0] for x in sections]
    exmetas=parse_braced_commands(masked,raw,'exercice');exstarts=[x[0] for x in exmetas]
    stm=parse_role_blocks(raw,masked,'statement');hints=parse_role_blocks(raw,masked,'hint');sols=parse_role_blocks(raw,masked,'solution')
    for s in stm:
        si=bisect.bisect_right(secstarts,s['start'])-1;s['section']=sections[si][1] if si>=0 else None
        ei=bisect.bisect_right(exstarts,s['start'])-1;s['source_metadata']=exmetas[ei][1] if ei>=0 else None
    byid=defaultdict(list)
    for s in stm: byid[s['exercise_id']].append(s)
    canonical={};nonident=[]
    for eid,occ in byid.items():
        groups=defaultdict(list)
        for x in occ: groups[(sha256_text(x['body']),x.get('title') or '')].append(x)
        if len(groups)>1: nonident.append(eid)
        for j,g in enumerate(groups.values(),1):
            exkey=eid if len(groups)==1 else f'{eid}::variant{j}';base=dict(g[0]);base['exercise_key']=exkey;base['sections']=list(dict.fromkeys(x['section'] for x in g if x.get('section')));base['duplicate_occurrences']=len(g);canonical[exkey]=base
    units=[];relations=[];stmt_unit={};stmt_ctx={};anomalies=[]
    def make_unit(exkey,role,x):
        sx=canonical[exkey]['sections'];h=sha256_text(x['body']);uid=stable_id('unit','exo7',exkey,role,h);code,name=split_section_title(sx[0] if sx else '')
        labels=[('exo7' if role=='statement' else 'ind7' if role=='hint' else 'cor7')+':'+x['exercise_id']]
        labels += [lab for lab in re.findall(r'\\label\{([^}]+)\}',x['body']) if lab not in labels]
        refs=re.findall(r'\\(?:ref|eqref|autoref|cref|Cref)\{([^}]+)\}',x['body']);cmds=sorted(c for c in extract_commands(x['body']) if c in macro_map)
        return {'unit_id':uid,'document_id':doc_id,'source_file':source_file,'kind':role,'unit_type':'exercise_'+role,'environment':None,'title':canonical[exkey].get('title') or None,'label':labels[0],'labels':labels,'latex':x['body'],'content_hash':h,'char_start':x['start'],'char_end':x['end'],'line_start':line_number(raw,x['start']),'line_end':line_number(raw,x['end']),'section_path':([{'kind':'section','title':sx[0]}] if sx else []),'section_aliases':sx,'section_code':code,'section_title':name,'proof_of':None,'related_unit_ids':[],'references':refs,'macros_used':cmds,'exercise_id':x['exercise_id'],'exercise_key':exkey,'exercise_role':role,'source_metadata':canonical[exkey].get('source_metadata') if role=='statement' else None,'source_anomalies':([x['anomaly']] if x.get('anomaly') else []),'requires_visual_context':('Figure externe omise' in x['body'])}
    for exkey,x in canonical.items():
        u=make_unit(exkey,'statement',x);units.append(u);stmt_unit[exkey]=u;stmt_ctx[exkey]=x['body']
        if x.get('anomaly'): anomalies.append({'exercise_key':exkey,'anomaly':x['anomaly']})
    for role,arr in [('hint',hints),('solution',sols)]:
        g=defaultdict(list)
        for x in arr:g[x['exercise_id']].append(x)
        for eid,occ in g.items():
            usable=[x for x in occ if x['body'].strip()]
            if not usable:
                continue
            uniq={sha256_text(x['body']):x for x in usable};matches=[k for k in canonical if k==eid or k.startswith(eid+'::')]
            for exkey in matches:
                for j,x in enumerate(uniq.values(),1):
                    u=make_unit(exkey,role,x);units.append(u);target=stmt_unit[exkey]['unit_id'];u['related_unit_ids']=[target]
                    typ='hint_of' if role=='hint' else 'solution_of';relations.append({'relation_id':stable_id('rel',typ,u['unit_id'],target),'type':typ,'from_unit_id':u['unit_id'],'to_unit_id':target,'document_id':doc_id})
    allby=defaultdict(list)
    for u in units:allby[u['exercise_key']].append(u['unit_id'])
    for u in units:u['related_unit_ids']=sorted(set(u['related_unit_ids']+[x for x in allby[u['exercise_key']] if x!=u['unit_id']]))
    label_occ=defaultdict(list)
    local_labelmap={}
    for u in units:
        for lab in u['labels']:
            label_occ[lab].append(u['unit_id'])
            local_labelmap[(u['exercise_key'],lab)]=u['unit_id']
    global_unique={lab:ids[0] for lab,ids in label_occ.items() if len(set(ids))==1}
    for u in units:
        for ref_index,ref in enumerate(u['references'],1):
            target=local_labelmap.get((u['exercise_key'],ref)) or global_unique.get(ref)
            relations.append({'relation_id':stable_id('rel','references',u['unit_id'],str(ref_index),ref,target or 'unresolved'),'type':'references','from_unit_id':u['unit_id'],'to_unit_id':target,'target_label':ref,'document_id':doc_id,'resolved':target is not None})
    chunks=[];forced=0;repaired=0
    for u in units:
        pieces,f=split_role(u['latex'],args.target_chars,args.max_chars);forced+=int(f);stnorm,n0=repair_embedding_text(stmt_ctx[u['exercise_key']]);secnorm,n1=repair_embedding_text(' | '.join(u['section_aliases']));titlenorm,n2=repair_embedding_text(u.get('title') or '');mctx=[macro_map[c] for c in u['macros_used'][:20] if c in macro_map]
        for i,piece in enumerate(pieces,1):
            norm,n3=repair_embedding_text(piece);rep=(n0+n1+n2+n3)>0;repaired+=int(rep);context=('\nStatement context:\n'+stnorm[:2200]) if u['exercise_role']!='statement' else ''
            emb=f"Corpus: Exo7 corrected mathematics exercises\nSection: {secnorm}\nType: {u['unit_type']}\nExercise ID: {u['exercise_id']}\nTitle: {titlenorm or '(untitled)'}\nRole: {u['exercise_role']}\nPart: {i}/{len(pieces)}{context}\nContent:\n{norm}"
            cid=stable_id('chunk',u['unit_id'],str(i),sha256_text(piece));chunks.append({'chunk_id':cid,'qdrant_id':qdrant_uuid(cid),'document_id':doc_id,'source_group':doc['source_group'],'course_title':doc['title'],'language':'fr','source_file':source_file,'unit_id':u['unit_id'],'unit_type':u['unit_type'],'content_kind':'exercise','exercise_id':u['exercise_id'],'exercise_key':u['exercise_key'],'exercise_role':u['exercise_role'],'title':u['title'],'label':u['label'],'labels':u['labels'],'section_path':u['section_path'],'section_aliases':u['section_aliases'],'section_path_text':' | '.join(u['section_aliases']),'section_code':u['section_code'],'section_title':u['section_title'],'proof_of':None,'related_unit_ids':u['related_unit_ids'],'references':u['references'],'macros_used':u['macros_used'],'macro_context':mctx,'part':i,'parts_total':len(pieces),'embedding_text':emb,'latex':piece,'content_hash':sha256_text(piece),'source_line_start':u['line_start'],'source_line_end':u['line_end'],'document_complete_inputs':True,'macro_context_complete':True,'requires_visual_context':u['requires_visual_context'],'source_anomalies':u['source_anomalies'],'embedding_text_encoding_repaired':rep})
    args.out.mkdir(parents=True,exist_ok=True)
    def write(name,rs):
        with (args.out/name).open('w',encoding='utf-8') as f:
            for r in rs:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    write('documents.jsonl',[doc]);write('macros.jsonl',macros);write('units.jsonl',units);write('relations.jsonl',relations);write('chunks.jsonl',chunks)
    solution_targets={r['to_unit_id'] for r in relations if r['type']=='solution_of'};hint_targets={r['to_unit_id'] for r in relations if r['type']=='hint_of'}
    stats={'source_chars':len(raw),'sections':len(sections),'statement_occurrences':len(stm),'canonical_exercises':len(canonical),'duplicate_statement_occurrences_merged':len(stm)-len(canonical),'hints':len(hints),'solutions':len(sols),'units':len(units),'chunks':len(chunks),'chunk_types':dict(Counter(c['unit_type'] for c in chunks)),'relations':dict(Counter(r['type'] for r in relations)),'max_chunk_latex_chars':max(len(c['latex']) for c in chunks),'forced_split_units':forced,'source_anomalies':anomalies,'chunks_with_embedding_encoding_repair':repaired,'visual_context_chunks':sum(c['requires_visual_context'] for c in chunks),'empty_source_solutions':sum(1 for x in sols if not x['body'].strip()),'statements_without_usable_solution':sum(u['unit_id'] not in solution_targets for u in stmt_unit.values()),'statements_without_hint':sum(u['unit_id'] not in hint_targets for u in stmt_unit.values()),'duplicate_nonidentical_ids':nonident}
    (args.out/'exercise_stats.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(stats,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
