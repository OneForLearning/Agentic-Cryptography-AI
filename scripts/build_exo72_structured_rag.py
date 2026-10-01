from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path

DOC_NAMESPACE = uuid.UUID('f62751b5-1412-4821-939c-a0544b60eac8')
COURSE_TITLE = 'Exo7 — corpus mathématique structuré (algèbre, analyse, géométrie, probabilités)'

ROLE_ENV = {
    'statement': 'enonceRAG',
    'hint': 'indicationRAG',
    'solution': 'correctionRAG',
}

FORMAL_ENVS = {
    'theoreme': 'theorem',
    'lemme': 'lemma',
    'proposition': 'proposition',
    'propriete': 'property',
    'corollaire': 'corollary',
    'definition': 'definition',
    'remarque': 'remark',
}

SEMANTIC_CUES = {
    'definition': re.compile(r'\b(?:d[ée]finition|definition)\b', re.I),
    'property': re.compile(r'\bpropri[ée]t[ée]\b', re.I),
    'proposition': re.compile(r'\bproposition\b', re.I),
    'lemma': re.compile(r'\blemme\b', re.I),
    'corollary': re.compile(r'\bcorollaire\b', re.I),
    'theorem': re.compile(r'\bth[ée]or[èe]me\b', re.I),
    'remark': re.compile(r'\bremarque\b', re.I),
    'hypothesis': re.compile(r'\bhypoth[èe]se\b', re.I),
    'conclusion': re.compile(r'\bconclusion\b|\bconclu(?:re|ons|sion)\b', re.I),
}

PROOF_REQUEST_RE = re.compile(r'\b(?:montrer|d[ée]montrer|prouver|[ée]tablir|justifier)\b', re.I)

DISPLAY_ENV_RE = re.compile(
    r'\\begin\{(?P<env>equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?|alignat\*?)\}'
    r'(?P<body>.*?)\\end\{(?P=env)\}',
    re.S,
)

UNIT_RE = re.compile(
    r'% RAG-UNIT-BEGIN\s+exercise_id=(?P<eid>\d+)\s+role=exercise\s*'
    r'(?P<body>.*?)% RAG-UNIT-END',
    re.S,
)


def stable_id(prefix: str, *parts: str, n: int = 24) -> str:
    raw = '\x1f'.join(parts).encode('utf-8')
    return f'{prefix}-' + hashlib.sha256(raw).hexdigest()[:n]


def qdrant_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(DOC_NAMESPACE, chunk_id))


def canonical_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip())


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')


def extract_macros(preamble: str) -> list[dict]:
    out = []
    pat = re.compile(
        r'\\(?P<kind>newcommand|renewcommand|providecommand)\s*'
        r'\{?\\(?P<name>[A-Za-z@]+)\}?\s*'
        r'(?:\[(?P<nargs>\d+)\])?\s*'
        r'\{(?P<body>[^\n]*)\}'
    )
    for m in pat.finditer(preamble):
        out.append({
            'macro_id': stable_id('macro', m.group('name'), m.group('body')),
            'command': m.group('name'),
            'kind': m.group('kind'),
            'num_args': int(m.group('nargs') or 0),
            'definition': m.group('body').strip(),
        })
    return out


def extract_role(block: str, env: str, *, allow_recovery: bool = False) -> tuple[str | None, str | None, list[str]]:
    """Returns body, optional title, anomalies."""
    anomalies = []
    start = re.search(r'\\begin\{' + re.escape(env) + r'\}(?:\[(?P<title>[^\]]*)\])?', block, re.S)
    if not start:
        return None, None, anomalies
    title = start.group('title')
    end = re.search(r'\\end\{' + re.escape(env) + r'\}', block[start.end():], re.S)
    if end:
        return block[start.end(): start.end() + end.start()].strip(), title, anomalies
    if not allow_recovery:
        anomalies.append(f'missing_end_{env}')
        return None, title, anomalies
    # Conservative recovery: current RAG unit boundary is authoritative.
    anomalies.append(f'missing_end_{env}')
    tail = block[start.end():]
    stop_positions = []
    for other in ROLE_ENV.values():
        if other == env:
            continue
        mm = re.search(r'\\begin\{' + re.escape(other) + r'\}', tail)
        if mm:
            stop_positions.append(mm.start())
    stop = min(stop_positions) if stop_positions else len(tail)
    return tail[:stop].strip(), title, anomalies


def semantic_cues(text: str) -> list[str]:
    return sorted(name for name, pat in SEMANTIC_CUES.items() if pat.search(text))


def formal_objects(text: str, parent_unit_id: str, exercise_id: str, role: str) -> list[dict]:
    out = []
    for env, kind in FORMAL_ENVS.items():
        pat = re.compile(
            r'\\begin\{' + re.escape(env) + r'\}(?:\[(?P<title>[^\]]*)\])?'
            r'(?P<body>.*?)\\end\{' + re.escape(env) + r'\}',
            re.S,
        )
        for idx, m in enumerate(pat.finditer(text), 1):
            raw = m.group(0).strip()
            out.append({
                'formal_id': stable_id('formal', parent_unit_id, env, str(idx), raw),
                'parent_unit_id': parent_unit_id,
                'exercise_id': exercise_id,
                'role': role,
                'formal_type': kind,
                'source_environment': env,
                'title': (m.group('title') or '').strip() or None,
                'latex': raw,
                'content_hash': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
            })
    return out


def protect_math(text: str) -> tuple[str, dict[str, str]]:
    """Protect display formulas while creating split boundaries."""
    spans = []
    for m in DISPLAY_ENV_RE.finditer(text):
        spans.append((m.start(), m.end()))
    for pat in (re.compile(r'\$\$(.*?)\$\$', re.S), re.compile(r'\\\[(.*?)\\\]', re.S)):
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    spans.sort()
    # discard overlaps, preferring earliest/longest
    clean = []
    last = -1
    for s, e in sorted(spans, key=lambda x: (x[0], -(x[1]-x[0]))):
        if s >= last:
            clean.append((s, e)); last = e
    repl = {}
    pieces = []
    pos = 0
    for i, (s, e) in enumerate(clean):
        token = f'@@MATH_BLOCK_{i:05d}@@'
        pieces.append(text[pos:s])
        pieces.append(token)
        repl[token] = text[s:e]
        pos = e
    pieces.append(text[pos:])
    return ''.join(pieces), repl


def restore_math(text: str, repl: dict[str, str]) -> str:
    for token, value in repl.items():
        text = text.replace(token, value)
    return text


def split_long_solution(text: str, target: int = 4500, max_chars: int = 6000) -> tuple[list[str], bool]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text], False
    protected, repl = protect_math(text)
    # Strong boundaries: top-level-ish items and blank paragraphs. We keep the marker with the following segment.
    raw_parts = re.split(r'(?=\\item\b)|\n\s*\n+', protected)
    parts = [p.strip() for p in raw_parts if p.strip()]
    groups: list[str] = []
    cur = ''
    forced = False
    for part in parts:
        candidate = part if not cur else cur + '\n\n' + part
        if len(restore_math(candidate, repl)) <= target:
            cur = candidate
            continue
        if cur:
            groups.append(restore_math(cur, repl).strip())
            cur = ''
        if len(restore_math(part, repl)) <= max_chars:
            cur = part
            continue
        # Last resort: split the protected fragment by lines/sentence-like punctuation, never inside protected math.
        forced = True
        subparts = re.split(r'(?<=\.)\s+(?=[A-ZÀ-ÖØ-Þ\\])|\n+', part)
        subcur = ''
        for sub in [x.strip() for x in subparts if x.strip()]:
            cand = sub if not subcur else subcur + '\n' + sub
            if len(restore_math(cand, repl)) <= max_chars:
                subcur = cand
            else:
                if subcur:
                    groups.append(restore_math(subcur, repl).strip())
                # If even one atomic subpart is huge, retain it intact rather than cut a formula/macro expression.
                subcur = sub
        if subcur:
            groups.append(restore_math(subcur, repl).strip())
    if cur:
        groups.append(restore_math(cur, repl).strip())
    return groups, forced


def extract_formulas(chunk_latex: str, chunk_id: str, unit_id: str, exercise_id: str, role: str) -> list[dict]:
    matches = []
    for m in DISPLAY_ENV_RE.finditer(chunk_latex):
        matches.append((m.start(), m.end(), m.group(0), m.group('env')))
    for pat, kind in ((re.compile(r'\$\$(.*?)\$\$', re.S), 'double_dollar'), (re.compile(r'\\\[(.*?)\\\]', re.S), 'bracket_display')):
        for m in pat.finditer(chunk_latex):
            matches.append((m.start(), m.end(), m.group(0), kind))
    # non-overlapping canonical ordering
    matches.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    out=[]; last=-1; idx=0
    for s,e,raw,kind in matches:
        if s < last:
            continue
        last=e; idx += 1
        labels = re.findall(r'\\label\{([^}]+)\}', raw)
        tags = re.findall(r'\\tag\{([^}]+)\}', raw)
        out.append({
            'formula_id': stable_id('formula', chunk_id, str(idx), raw),
            'chunk_id': chunk_id,
            'unit_id': unit_id,
            'exercise_id': exercise_id,
            'role': role,
            'formula_kind': kind,
            'latex': raw.strip(),
            'labels': labels,
            'tags': tags,
            'content_hash': hashlib.sha256(raw.strip().encode('utf-8')).hexdigest(),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Build a graph-aware RAG dataset from the sanitized Exo7 v2 LaTeX corpus.')
    ap.add_argument('source', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--target-chars', type=int, default=4500)
    ap.add_argument('--max-chars', type=int, default=6000)
    args = ap.parse_args()

    raw = args.source.read_text(encoding='utf-8')
    preamble = raw.split('\\begin{document}', 1)[0]
    macros = extract_macros(preamble)
    document_id = stable_id('doc', str(args.source.name), hashlib.sha256(raw.encode('utf-8')).hexdigest())

    documents = [{
        'document_id': document_id,
        'course_title': COURSE_TITLE,
        'title': 'Exo7 — corpus d’exercices mathématiques structuré',
        'source_file': str(args.source),
        'source_group': 'exo7_structured_v2',
        'language': 'fr',
        'source_chars': len(raw),
        'source_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
        'rag_contract': 'exercise statement/hint/solution + topics + compressed related clusters + formula graph',
    }]

    unit_blocks = UNIT_RE.findall(raw)
    if not unit_blocks:
        raise RuntimeError('Aucune unité RAG-UNIT-BEGIN/RAG-UNIT-END détectée.')

    units=[]; chunks=[]; relations=[]; formulas=[]; formula_relations=[]; quality_flags=[]; formal_records=[]
    topic_members=defaultdict(list); cluster_members_map={}; duplicate_map=defaultdict(list)
    exercise_statement_unit={}; exercise_role_units=defaultdict(dict)
    old_label_owner={}; unresolved_ref_records=[]
    source_anomaly_count=Counter(); forced_split_units=0

    # First pass: role units and chunks.
    for eid, block in unit_blocks:
        topic_line = re.search(r'^% RAG-TOPICS:\s*(.*)$', block, re.M)
        topics = [x.strip() for x in (topic_line.group(1).split('|') if topic_line else []) if x.strip()]
        for t in topics:
            topic_members[t].append(eid)

        related_line = re.search(r'^% RAG-RELATED:\s*(.*)$', block, re.M)
        related_cluster_id = None
        if related_line:
            members = sorted(set([eid] + [x.strip() for x in related_line.group(1).split(',') if x.strip()]))
            related_cluster_id = stable_id('cluster', *members, n=20)
            cluster_members_map.setdefault(related_cluster_id, members)

        stmt_body, stmt_title, stmt_anom = extract_role(block, 'enonceRAG', allow_recovery=True)
        role_data=[]
        if stmt_body is not None:
            role_data.append(('statement', stmt_body, stmt_title, stmt_anom))
        else:
            source_anomaly_count['missing_statement'] += 1
            quality_flags.append({
                'quality_flag_id': stable_id('qf', eid, 'missing_statement'),
                'exercise_id': eid,
                'unit_id': None,
                'chunk_id': None,
                'flag': 'missing_statement',
                'severity': 'critical',
                'detail': 'Aucun environnement enonceRAG récupérable dans l’unité source.',
            })

        for role, env in [('hint','indicationRAG'),('solution','correctionRAG')]:
            body,title,anom = extract_role(block, env, allow_recovery=False)
            if body is not None and body.strip():
                role_data.append((role,body,title,anom))

        for role, body, title, anomalies in role_data:
            unit_id = stable_id('unit', document_id, eid, role)
            exercise_role_units[eid][role] = unit_id
            if role == 'statement': exercise_statement_unit[eid]=unit_id
            cues = semantic_cues(body)
            labels = re.findall(r'\\label\{([^}]+)\}', body)
            refs = re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}', body)
            for lab in labels:
                old_label_owner.setdefault(lab, unit_id)

            visual = '[Figure externe omise]' in body or '\\begin{tikzpicture}' in body or '\\begin{asy}' in body
            odd_dollars = body.count('$$') % 2 == 1
            qflags=[]
            for a in anomalies:
                source_anomaly_count[a]+=1; qflags.append(a)
            if odd_dollars:
                source_anomaly_count['unbalanced_double_dollar'] += 1; qflags.append('unbalanced_double_dollar')
            if visual:
                qflags.append('requires_visual_context')

            proof_like = bool(role == 'solution' and stmt_body and PROOF_REQUEST_RE.search(stmt_body))
            formal = formal_objects(body, unit_id, eid, role)
            formal_records.extend(formal)
            formal_types=sorted(set(x['formal_type'] for x in formal))

            unit = {
                'unit_id': unit_id,
                'document_id': document_id,
                'exercise_id': eid,
                'unit_type': 'exercise_' + role,
                'content_kind': 'exercise',
                'role': role,
                'semantic_role': 'proof_solution' if proof_like else role,
                'title': title or f'Exercice {eid}',
                'topics': topics,
                'related_cluster_id': related_cluster_id,
                'semantic_cues': cues,
                'embedded_formal_types': formal_types,
                'labels': labels,
                'refs': refs,
                'requires_visual_context': visual,
                'quality_flags': qflags,
                'requires_human_math_check': any(x in qflags for x in ['missing_end_enonceRAG','unbalanced_double_dollar']),
                'latex': body.strip(),
                'content_hash': hashlib.sha256(body.strip().encode('utf-8')).hexdigest(),
            }
            units.append(unit)
            for flag in qflags:
                severity = 'warning' if flag == 'requires_visual_context' else 'major'
                quality_flags.append({
                    'quality_flag_id': stable_id('qf', unit_id, flag),
                    'exercise_id': eid,
                    'unit_id': unit_id,
                    'chunk_id': None,
                    'flag': flag,
                    'severity': severity,
                    'detail': {
                        'missing_end_enonceRAG':'Environnement enonceRAG sans fin explicite; contenu récupéré jusqu’à la frontière RAG-UNIT-END.',
                        'unbalanced_double_dollar':'Nombre impair de délimiteurs $$ dans l’unité source.',
                        'requires_visual_context':'Le contenu dépend d’une figure externe omise ou d’un environnement graphique.',
                    }.get(flag, flag),
                })

            if role == 'solution':
                parts, forced = split_long_solution(body, args.target_chars, args.max_chars)
                if forced: forced_split_units += 1
            else:
                parts = [body.strip()]

            statement_context = ''
            if role != 'statement' and stmt_body:
                statement_context = canonical_text(stmt_body)[:2200]

            for pi, part in enumerate(parts, 1):
                chunk_id = stable_id('chunk', unit_id, str(pi), part)
                section_path_text = ' > '.join(topics)
                emb_lines = [
                    'Corpus: Exo7 mathematical exercises (structured v2)',
                    f'Exercise ID: {eid}',
                    f'Role: {role}',
                    f'Semantic role: {"proof_solution" if proof_like else role}',
                ]
                if title:
                    emb_lines.append(f'Title: {canonical_text(title)}')
                if topics:
                    emb_lines.append('Topics: ' + ' | '.join(topics))
                if cues:
                    emb_lines.append('Semantic cues in source: ' + ', '.join(cues))
                if formal_types:
                    emb_lines.append('Embedded formal objects: ' + ', '.join(formal_types))
                if len(parts)>1:
                    emb_lines.append(f'Part: {pi}/{len(parts)}')
                if statement_context:
                    emb_lines.extend(['Statement context:', statement_context])
                emb_lines.extend(['Content:', canonical_text(part)])
                chunk = {
                    'chunk_id': chunk_id,
                    'qdrant_id': qdrant_uuid(chunk_id),
                    'unit_id': unit_id,
                    'document_id': document_id,
                    'exercise_id': eid,
                    'course_title': COURSE_TITLE,
                    'source_file': str(args.source),
                    'source_group': 'exo7_structured_v2',
                    'language': 'fr',
                    'unit_type': 'exercise_' + role,
                    'content_kind': 'exercise',
                    'role': role,
                    'semantic_role': 'proof_solution' if proof_like else role,
                    'title': title or f'Exercice {eid}',
                    'section_path': [{'kind':'topic','title':t} for t in topics],
                    'section_path_text': section_path_text,
                    'topics': topics,
                    'related_cluster_id': related_cluster_id,
                    'semantic_cues': cues,
                    'embedded_formal_types': formal_types,
                    'part_index': pi,
                    'part_count': len(parts),
                    'proof_of': None,
                    'related_unit_ids': [],
                    'labels': re.findall(r'\\label\{([^}]+)\}', part),
                    'refs': re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}', part),
                    'requires_visual_context': visual,
                    'quality_flags': qflags,
                    'requires_human_math_check': unit['requires_human_math_check'],
                    'embedding_text': '\n'.join(emb_lines),
                    'latex': part.strip(),
                    'content_hash': hashlib.sha256(part.strip().encode('utf-8')).hexdigest(),
                }
                chunks.append(chunk)
                duplicate_key_hash = hashlib.sha256(canonical_text(part).encode('utf-8')).hexdigest()
                duplicate_map[(role, duplicate_key_hash)].append(chunk_id)
                fms = extract_formulas(part, chunk_id, unit_id, eid, role)
                formulas.extend(fms)
                for fm in fms:
                    formula_relations.append({
                        'formula_relation_id': stable_id('frel', fm['formula_id'], chunk_id),
                        'type': 'formula_in_chunk',
                        'formula_id': fm['formula_id'],
                        'chunk_id': chunk_id,
                        'unit_id': unit_id,
                    })

    # Same-exercise role relations and related unit ids.
    unit_by_id={u['unit_id']:u for u in units}
    chunks_by_unit=defaultdict(list)
    for c in chunks: chunks_by_unit[c['unit_id']].append(c)
    for eid, roles in exercise_role_units.items():
        st=roles.get('statement')
        for role, default_rtype in [('hint','hint_of'),('solution','solution_of')]:
            uid=roles.get(role)
            if st and uid:
                rtype = default_rtype
                if role == 'solution' and unit_by_id.get(uid, {}).get('semantic_role') == 'proof_solution':
                    rtype = 'proof_solution_of'
                rid=stable_id('rel', rtype, uid, st, n=32)
                relations.append({'relation_id':rid,'type':rtype,'from_unit_id':uid,'to_unit_id':st,'exercise_id':eid,'resolved':True})
                for c in chunks_by_unit[uid]: c['related_unit_ids'].append(st)
                for c in chunks_by_unit[st]: c['related_unit_ids'].append(uid)

    # Resolve explicit refs to units.
    for u in units:
        for ref in u.get('refs',[]):
            target=old_label_owner.get(ref)
            relations.append({
                'relation_id': stable_id('rel','explicit_ref',u['unit_id'],ref,target or '',n=32),
                'type':'explicit_ref','from_unit_id':u['unit_id'],'to_unit_id':target,
                'target_label':ref,'resolved':bool(target),
            })
            if not target:
                unresolved_ref_records.append({'from_unit_id':u['unit_id'],'target_label':ref})
            else:
                for c in chunks_by_unit[u['unit_id']]:
                    if target not in c['related_unit_ids']: c['related_unit_ids'].append(target)

    # Formal objects are linked to their parent role unit; theorem-like objects remain discoverable in metadata.
    for f in formal_records:
        relations.append({
            'relation_id': stable_id('rel','contains_formal',f['parent_unit_id'],f['formal_id'],n=32),
            'type':'contains_formal','from_unit_id':f['parent_unit_id'],'to_unit_id':None,
            'formal_id':f['formal_id'],'formal_type':f['formal_type'],'resolved':True,
        })

    # Remove exact duplicate relations (e.g. repeated \ref to the same target within one unit).
    _seen_rel = set()
    _dedup_rel = []
    for _r in relations:
        _key = json.dumps({k:v for k,v in _r.items() if k != 'relation_id'}, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        if _key in _seen_rel:
            continue
        _seen_rel.add(_key)
        _r = dict(_r)
        _r['relation_id'] = stable_id('rel', _key, n=32)
        _dedup_rel.append(_r)
    relations = _dedup_rel

    topics=[]
    for topic, members in sorted(topic_members.items()):
        topics.append({'topic_id':stable_id('topic',topic,n=20),'topic':topic,'member_exercise_ids':sorted(set(members)),'member_count':len(set(members))})

    exercise_topics=[]
    for topic, members in sorted(topic_members.items()):
        tid=stable_id('topic',topic,n=20)
        for eid in sorted(set(members)):
            exercise_topics.append({'exercise_topic_id':stable_id('etop',eid,tid,n=24),'exercise_id':eid,'topic_id':tid,'topic':topic})

    semantic_annotations=[]
    for u in units:
        for cue in u.get('semantic_cues',[]):
            semantic_annotations.append({'semantic_annotation_id':stable_id('sem',u['unit_id'],cue,n=24),'unit_id':u['unit_id'],'exercise_id':u['exercise_id'],'cue':cue,'source':'lexical_cue','is_formal':False})
    for f in formal_records:
        semantic_annotations.append({'semantic_annotation_id':stable_id('sem',f['formal_id'],f['formal_type'],n=24),'unit_id':f['parent_unit_id'],'exercise_id':f['exercise_id'],'cue':f['formal_type'],'source':'latex_environment','is_formal':True,'formal_id':f['formal_id']})

    related_clusters=[]
    for cid,members in sorted(cluster_members_map.items()):
        related_clusters.append({'related_cluster_id':cid,'member_exercise_ids':members,'member_count':len(members)})

    duplicate_groups=[]
    for (role,h), ids in duplicate_map.items():
        if len(ids)>1:
            duplicate_groups.append({'duplicate_group_id':stable_id('dup',role,h,n=20),'role':role,'content_hash':h,'chunk_ids':ids,'count':len(ids)})

    # Add duplicate group ids to chunks (primarily useful for exact duplicate statements).
    dup_by_chunk={}
    for g in duplicate_groups:
        for cid in g['chunk_ids']: dup_by_chunk[cid]=g['duplicate_group_id']
    for c in chunks:
        c['duplicate_group_id']=dup_by_chunk.get(c['chunk_id'])

    # Quality flags attach to all relevant chunks.
    q_by_unit=defaultdict(list)
    for q in quality_flags:
        if q.get('unit_id'): q_by_unit[q['unit_id']].append(q['flag'])
    for c in chunks:
        c['quality_flags']=sorted(set(c['quality_flags'] + q_by_unit.get(c['unit_id'],[])))

    out=args.out
    if out.exists():
        import shutil; shutil.rmtree(out)
    out.mkdir(parents=True)
    write_jsonl(out/'documents.jsonl', documents)
    write_jsonl(out/'macros.jsonl', macros)
    write_jsonl(out/'units.jsonl', units)
    write_jsonl(out/'chunks.jsonl', chunks)
    write_jsonl(out/'relations.jsonl', relations)
    write_jsonl(out/'formulas.jsonl', formulas)
    write_jsonl(out/'formula_relations.jsonl', formula_relations)
    write_jsonl(out/'formal_objects.jsonl', formal_records)
    write_jsonl(out/'topics.jsonl', topics)
    write_jsonl(out/'exercise_topics.jsonl', exercise_topics)
    write_jsonl(out/'semantic_annotations.jsonl', semantic_annotations)
    write_jsonl(out/'related_clusters.jsonl', related_clusters)
    write_jsonl(out/'duplicate_groups.jsonl', duplicate_groups)
    write_jsonl(out/'quality_flags.jsonl', quality_flags)
    write_jsonl(out/'unresolved_refs.jsonl', unresolved_ref_records)

    role_counts=Counter(u['unit_type'] for u in units)
    chunk_counts=Counter(c['unit_type'] for c in chunks)
    q_counts=Counter(q['flag'] for q in quality_flags)
    stats={
        'source_chars':len(raw),
        'source_sha256':documents[0]['source_sha256'],
        'exercise_units':len(unit_blocks),
        'units':len(units),
        'chunks':len(chunks),
        'unit_types':dict(role_counts),
        'chunk_types':dict(chunk_counts),
        'formulas':len(formulas),
        'formal_objects':len(formal_records),
        'formal_types':dict(Counter(f['formal_type'] for f in formal_records)),
        'topics':len(topics),
        'topic_memberships':len(exercise_topics),
        'semantic_annotations':len(semantic_annotations),
        'proof_like_solution_units':sum(1 for u in units if u.get('semantic_role')=='proof_solution'),
        'titled_exercises':sum(1 for u in units if u['role']=='statement' and u['title'] != f"Exercice {u['exercise_id']}"),
        'semantic_cue_counts':dict(Counter(cue for u in units for cue in u.get('semantic_cues',[]))),
        'exercises_with_topics':sum(1 for u in units if u['role']=='statement' and u['topics']),
        'related_clusters':len(related_clusters),
        'relations':len(relations),
        'relation_types':dict(Counter(r['type'] for r in relations)),
        'labels':sum(len(u['labels']) for u in units),
        'unique_labels':len(old_label_owner),
        'refs':sum(len(u['refs']) for u in units),
        'unresolved_refs':len(unresolved_ref_records),
        'duplicate_groups':len(duplicate_groups),
        'duplicate_statement_groups':sum(1 for g in duplicate_groups if g['role']=='statement'),
        'duplicate_statement_occurrences':sum(g['count'] for g in duplicate_groups if g['role']=='statement'),
        'quality_flags':len(quality_flags),
        'quality_flag_types':dict(q_counts),
        'forced_split_units':forced_split_units,
        'max_chunk_latex_chars':max(len(c['latex']) for c in chunks),
        'chunks_over_max_chars':sum(len(c['latex'])>args.max_chars for c in chunks),
        'visual_context_chunks':sum(c['requires_visual_context'] for c in chunks),
        'source_anomalies':dict(source_anomaly_count),
    }
    (out/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
