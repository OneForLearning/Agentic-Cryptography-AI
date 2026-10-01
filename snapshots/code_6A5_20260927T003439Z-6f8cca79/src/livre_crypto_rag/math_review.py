"""Préparation de revue : localisation PDF et segmentation candidate, sans certification."""
import re
import unicodedata

KINDS = {
    'lemma': 'lemma', 'lemme': 'lemma', 'theorem': 'theorem', 'theoreme': 'theorem',
    'corollary': 'corollary', 'corollaire': 'corollary', 'definition': 'definition',
    'proposition': 'proposition', 'conjecture': 'conjecture', 'hypothesis': 'hypothesis',
    'hypothese': 'hypothesis', 'remark': 'remark', 'remarque': 'remark',
    'proof': 'proof', 'preuve': 'proof', 'demonstration': 'proof',
}
RESULTS = {'lemma', 'theorem', 'corollary', 'proposition', 'conjecture'}


def folded(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c)).lower()


def heading(line):
    clean = folded(line.strip())
    # Missing ligatures such as "De nition" are deliberately not guessed.
    m = re.match(r'^(lemma|lemme|theorem|theoreme|corollary|corollaire|definition|proposition|conjecture|hypothesis|hypothese|remark|remarque)\s+(\d+)\b', clean)
    if m:
        return KINDS[m[1]], m[2]
    if re.match(r'^(proof|preuve|demonstration)(?:\s*\(|\s*\.|\s*:)', clean):
        return 'proof', None
    if re.match(r'^\d+\s+Pascal Paillier\s*$', line.strip()):
        return None
    if re.match(r'^\d+(?:\.\d+)*\s+[A-ZÀ-ÖØ-Þ]', line.strip()):
        return 'section', None
    return None


def build_units(pages, run_id):
    """All text is kept; boundaries and proof links are hypotheses for human review."""
    units = []
    sources = sorted({p['source_id'] for p in pages})
    for source in sources:
        current = None
        for page in sorted((p for p in pages if p['source_id'] == source), key=lambda p: p['page']):
            offset = 0
            for line in page['text'].splitlines(keepends=True):
                label = heading(line)
                if current is None or label:
                    kind, number = label or ('context', None)
                    current = {
                        'unit_id': f'{run_id}:{source}:u{len(units):04d}',
                        'source_id': source, 'kind': kind, 'number': number,
                        'heading': line.strip() if label else '', 'text': '', 'segments': [],
                        'validation_status': 'NOT_REVIEWED', 'rag_admission': 'BLOCKED',
                        'boundary_status': 'CANDIDATE', 'formula_fidelity': 'NOT_REVIEWED',
                        'mathematical_validity': 'NOT_REVIEWED',
                    }
                    units.append(current)
                current['text'] += line
                segments = current['segments']
                end = offset + len(line)
                if segments and segments[-1]['page'] == page['page'] and segments[-1]['end'] == offset:
                    segments[-1]['end'] = end
                else:
                    segments.append({'page': page['page'], 'start': offset, 'end': end})
                offset = end
        local = [u for u in units if u['source_id'] == source]
        previous = None
        for u in local:
            u['pages'] = sorted({s['page'] for s in u['segments']})
            if u['kind'] == 'section':
                previous = None
            if u['kind'] in RESULTS:
                previous = u
            if u['kind'] == 'proof':
                explicit = re.match(r'^(?:proof|preuve|demonstration)\s*\(\s*(?:of |du |de |de la )?(lemma|lemme|theorem|theoreme|proposition|corollary|corollaire)\s+(\d+)\s*\)', folded(u['heading']))
                if explicit:
                    targets = [x for x in local if x['kind'] == KINDS[explicit[1]] and x['number'] == explicit[2]]
                    target = targets[0] if len(targets) == 1 else None
                    basis = 'explicit_label'
                else:
                    target = previous
                    basis = 'preceding_result'
                u['proof_of_candidate'] = target['unit_id'] if target else None
                u['link_basis'] = basis
                u['link_status'] = 'NOT_REVIEWED'
    return units


def assert_coverage(pages, units):
    """Check contiguous, exact coverage of every selected text, including whitespace."""
    for p in pages:
        segments = sorted((s for u in units if u['source_id'] == p['source_id']
                           for s in u['segments'] if s['page'] == p['page']), key=lambda s: s['start'])
        end = 0
        for s in segments:
            if s['start'] != end or s['end'] <= end:
                raise ValueError('Chevauchement ou perte de texte dans la segmentation')
            end = s['end']
        if end != len(p['text']):
            raise ValueError('Couverture incomplète')
