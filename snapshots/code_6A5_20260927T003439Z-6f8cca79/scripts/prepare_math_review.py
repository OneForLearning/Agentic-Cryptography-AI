"""Crée un nouveau run de revue visuelle depuis une extraction vérifiée, sans admission."""
import argparse
import html
import json
from pathlib import Path
import pymupdf
from livre_crypto_rag.extraction_runs import ExtractionRun, verify_run, sha256_file, write_json, write_jsonl
from livre_crypto_rag.math_review import build_units, assert_coverage

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_run', type=Path)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    origin = args.input_run.resolve()
    manifest = verify_run(origin)
    manifest_hash = sha256_file(origin / 'manifest.json')
    pages_file = origin / 'selected_pages.jsonl'
    if not pages_file.is_file():
        raise SystemExit('Une extraction du comparateur avec selected_pages.jsonl est requise.')
    # Refuse PDFs or catalogue changed since the extraction.
    for name, expected in manifest['input_sha256'].items():
        if sha256_file(ROOT / name) != expected:
            raise SystemExit(f'Entrée différente de celle du run : {name}')
    pages = [json.loads(line) for line in pages_file.read_text(encoding='utf-8').splitlines()]
    keys = [(p['source_id'], p['page']) for p in pages]
    if len(keys) != len(set(keys)):
        raise ValueError('Pages dupliquées')
    source_ids = {s['id'] for s in manifest['sources']}
    if set(s for s, _ in keys) != source_ids:
        raise ValueError('Sources incohérentes dans les pages')
    run = ExtractionRun(ROOT, manifest['sources'], 'math-review-candidates-v1',
                        {'origin_run_id': manifest['run_id'], 'origin_manifest_sha256': manifest_hash,
                         'segmentation_module_sha256': sha256_file(ROOT / 'src/livre_crypto_rag/math_review.py')},
                        __file__, args.run_id)
    (run.path / 'origin_manifest.json').write_bytes((origin / 'manifest.json').read_bytes())
    # Keep exact text, replace inherited claims of mathematical reliability.
    safe_pages = []
    for p in pages:
        safe_pages.append({
            'source_id': p['source_id'], 'page': p['page'], 'text': p['text'],
            'extractor': p['extractor'], 'text_sha256': __import__('hashlib').sha256(p['text'].encode()).hexdigest(),
            'validation_status': 'NOT_REVIEWED', 'rag_admission': 'BLOCKED',
            'formula_fidelity': 'NOT_REVIEWED', 'mathematical_validity': 'NOT_REVIEWED',
            'math_reliable': False, 'requires_human_source_check': True,
        })
    write_jsonl(run.path / 'review_pages.jsonl', safe_pages)
    units = build_units(safe_pages, run.run_id)
    assert_coverage(safe_pages, units)
    write_jsonl(run.path / 'units.jsonl', units)
    spatial = []
    images = run.path / 'images'
    images.mkdir()
    for source in manifest['sources']:
        with pymupdf.open(ROOT / source['file']) as pdf:
            for p in [p for p in safe_pages if p['source_id'] == source['id']]:
                page = pdf[p['page'] - 1]
                name = f"{source['id']}-p{p['page']:04d}.png"
                page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(images / name)
                # Each span retains its position and superscript flag; no inferred LaTeX.
                spatial.append({'source_id': source['id'], 'page': p['page'],
                                'source_sha256': run.source_hash(source),
                                'image': 'images/' + name, 'page_rect': list(page.rect),
                                'coordinate_system': 'PyMuPDF points, top-left, unrotated text coordinates',
                                'page_rotation': page.rotation, 'cropbox': list(page.cropbox),
                                'blocks': page.get_text('dict', flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)['blocks']})
    write_jsonl(run.path / 'spatial_pages.jsonl', spatial)
    # Page-specific proposals are used only when their PDF hash matches.
    proposals_file = ROOT / 'scripts/math_review_proposals.json'
    proposals = []
    if proposals_file.is_file():
        for item in json.loads(proposals_file.read_text(encoding='utf-8')):
            if manifest['input_sha256'].get(item['file']) == item['source_sha256']:
                proposals.append(item)
    write_json(run.path / 'correction_proposals.json', proposals)
    esc = lambda x: html.escape(str(x), quote=True)
    parts = ['<!doctype html><html lang="fr"><meta charset="utf-8"><title>Revue mathématique</title>',
             '<style>body{font:16px system-ui;margin:24px;max-width:1600px} .pair{display:grid;grid-template-columns:1fr 1fr;gap:20px} img{width:100%} pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px} article{border-top:2px solid #999;margin-top:30px} .warning{background:#fff0cf;padding:16px} @media(max-width:850px){.pair{display:block}}</style>',
             '<h1>Revue mathématique — propositions non validées</h1>',
             '<p class="warning">Aucune admission au RAG. Le texte de droite est inchangé. Les images permettent de vérifier les formules ; les coordonnées détaillées sont dans spatial_pages.jsonl. Les unités et liens sont des candidats à relire. Les propositions ne modifient pas le PDF.</p>',
             '<h2>Unités candidates et liens de preuve</h2><ul>']
    for u in units:
        target = u.get('proof_of_candidate')
        parts.append(f"<li>{esc(u['unit_id'])} — {esc(u['kind'])} {esc(u['number'] or '')}, pages {esc(u['pages'])}" + (f" → {esc(target)} ({esc(u['link_basis'])})" if target else '') + '</li>')
    parts.append('</ul>')
    for p, geo in zip(safe_pages, sorted(spatial, key=lambda x: keys.index((x['source_id'], x['page'])))):
        parts.append(f"<article><h2>{esc(p['source_id'])} — page {p['page']}</h2>")
        for item in proposals:
            if item['source_id'] == p['source_id'] and item['page'] == p['page']:
                parts.append(f"<p class='warning'><b>{esc(item['kind'])}</b> : {esc(item['observation'])}<br>Proposition : {esc(item['proposal'])}</p>")
        parts.append(f"<div class='pair'><img loading='lazy' src='{esc(geo['image'])}' alt='Page du PDF'><pre>{esc(p['text'])}</pre></div></article>")
    parts.append('</html>')
    (run.path / 'review.html').write_text('\n'.join(parts), encoding='utf-8')
    # Detect changes to the extraction while preparing review.
    verify_run(origin)
    if sha256_file(origin / 'manifest.json') != manifest_hash:
        raise RuntimeError('Manifeste source modifié pendant préparation')
    run.finish({'sources': len(manifest['sources']), 'pages': len(safe_pages), 'units': len(units),
                'proof_links_candidate': sum(bool(u.get('proof_of_candidate')) for u in units),
                'proposals': len(proposals), 'admitted_units': 0})
    print('Revue préparée :', run.path / 'review.html')
    print('Toutes les unités restent NOT_REVIEWED et BLOCKED.')


if __name__ == '__main__':
    main()
