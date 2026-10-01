"""Extraction pypdf témoin, sorties exclusivement sous rag/runs/<run-id>."""
import argparse
import re
from pathlib import Path
from pypdf import PdfReader
from livre_crypto_rag.extraction_runs import (
    ExtractionRun, select_sources, validate_chunk_parameters, write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    text = text.replace('\x00', '')
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def split_text(text, size, overlap):
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        part = text[start:end].strip()
        if part:
            yield part
        if end == len(text):
            break
        start = end - overlap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', action='append', default=[])
    p.add_argument('--run-id')
    p.add_argument('--chunk-size', type=int, default=4000)
    p.add_argument('--overlap', type=int, default=400)
    args = p.parse_args()
    validate_chunk_parameters(args.chunk_size, args.overlap)
    sources = select_sources(ROOT, args.source)
    run = ExtractionRun(ROOT, sources, 'pypdf-fixed-page-baseline',
                        {'chunk_size_characters': args.chunk_size, 'overlap_characters': args.overlap,
                         'min_page_characters': 100}, __file__, args.run_id)
    pages, chunks = [], []
    for source in sources:
        for number, page in enumerate(PdfReader(ROOT / source['file']).pages, 1):
            raw_text = page.extract_text() or ''
            text = normalize(raw_text)
            skipped = len(text) < 100
            pages.append({'source_id': source['id'], 'page': number, 'raw_text': raw_text,
                          'text': text, 'skipped_for_chunking': skipped,
                          'skip_reason': 'less_than_100_characters' if skipped else None,
                          'source_sha256': run.source_hash(source)})
            if skipped:
                continue
            for index, part in enumerate(split_text(text, args.chunk_size, args.overlap)):
                chunks.append({'chunk_id': run.chunk_id(source['id'], number, index),
                               'run_id': run.run_id, 'source_id': source['id'], 'page': number,
                               'chunk_index': index, 'domain': source['domain'], 'level': source['level'],
                               'topics': source['topics'], 'text': part, 'extractor': 'pypdf',
                               'source_sha256': run.source_hash(source),
                               'validation_status': 'NOT_REVIEWED'})
    write_jsonl(run.path / 'pages.jsonl', pages)
    write_jsonl(run.path / 'chunks.jsonl', chunks)
    counts = {'sources': len(sources), 'pages': len(pages), 'chunks': len(chunks),
              'skipped_pages': sum(p['skipped_for_chunking'] for p in pages)}
    run.finish(counts)
    print('Extraction COMPLETE :', run.path)
    print(counts)


if __name__ == '__main__':
    main()
