"""Sorties d'extraction versionnées ; aucune écriture dans corpus/processed."""
import hashlib
import json
import re
import sys
import platform
import subprocess
import shutil
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from uuid import uuid4

from .catalog import load_catalog


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def select_sources(root, wanted=()):
    catalog = load_catalog(root / 'corpus/catalog/sources.yaml', require_files=True, root=root)
    wanted = set(wanted)
    unknown = wanted - {s.id for s in catalog.sources}
    if unknown:
        raise ValueError(f'Sources inconnues : {sorted(unknown)}')
    selected = [s for s in catalog.sources if not wanted or s.id in wanted]
    if not selected:
        raise ValueError('Catalogue ou sélection vide')
    return [s.model_dump() for s in selected]


def validate_chunk_parameters(size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('chunk-size doit être > 0 et 0 <= overlap < chunk-size')


class ExtractionRun:
    def __init__(self, root, sources, method, parameters, script, run_id=None):
        self.root = Path(root).resolve()
        self.run_id = run_id or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                                 + '-' + uuid4().hex[:12])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', self.run_id):
            raise ValueError('run-id invalide')
        self.path = self.root / 'rag/runs' / self.run_id
        if not self.path.resolve().is_relative_to(self.root):
            raise ValueError('Sortie hors du projet')
        versions = {}
        for name in ('pypdf', 'pymupdf', 'pydantic', 'pyyaml'):
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                versions[name] = None
        executable = shutil.which('pdftotext')
        if executable:
            result = subprocess.run([executable, '-v'], capture_output=True, text=True, timeout=10)
            versions['pdftotext'] = (result.stderr or result.stdout).splitlines()[0]
        inputs = {}
        for source in sources:
            name = source['file']
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root / 'corpus/raw'):
                raise ValueError('PDF hors corpus/raw')
            inputs[name] = sha256_file(path)
        inputs['corpus/catalog/sources.yaml'] = sha256_file(self.root / 'corpus/catalog/sources.yaml')
        self.manifest = {
            'schema': 'ace-extraction-run-v1', 'run_id': self.run_id, 'status': 'RUNNING',
            'method': method, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'platform': platform.platform(), 'versions': versions,
            'parameters': parameters, 'sources': sources, 'input_sha256': inputs,
            'script_sha256': sha256_file(script),
            'helper_sha256': sha256_file(__file__),
            'catalog_module_sha256': sha256_file(Path(__file__).with_name('catalog.py')),
            'notice': 'Artefacts expérimentaux : aucune admission automatique au RAG.',
        }
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / 'catalog.yaml').write_bytes((self.root / 'corpus/catalog/sources.yaml').read_bytes())
        write_json(self.path / 'manifest.json', self.manifest)

    def source_hash(self, source):
        return self.manifest['input_sha256'][source['file']]

    def chunk_id(self, source_id, page, index):
        return f'{self.run_id}:{source_id}-p{page:04d}-c{index:03d}'

    def finish(self, counts):
        for name, expected in self.manifest['input_sha256'].items():
            if sha256_file(self.root / name) != expected:
                self.manifest['status'] = 'FAILED_INPUT_CHANGED'
                write_json(self.path / 'manifest.json', self.manifest)
                raise RuntimeError(f'Entrée modifiée pendant extraction : {name}')
        self.manifest['output_sha256'] = {
            str(p.relative_to(self.path)): sha256_file(p)
            for p in sorted(self.path.rglob('*')) if p.is_file() and p.name != 'manifest.json'
        }
        self.manifest['counts'] = counts
        self.manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
        self.manifest['status'] = 'COMPLETE'
        write_json(self.path / 'manifest.json', self.manifest)


def verify_run(path):
    path = Path(path).resolve()
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'ace-extraction-run-v1' or manifest.get('status') != 'COMPLETE':
        raise ValueError('Extraction incomplète ou format inconnu')
    expected = manifest['output_sha256']
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if set(expected) != actual:
        raise ValueError('Inventaire des sorties différent du manifeste')
    for name, checksum in expected.items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or sha256_file(target) != checksum:
            raise ValueError(f'Sortie modifiée : {name}')
    return manifest
