import json
import tempfile
import unittest
from pathlib import Path
import yaml
from livre_crypto_rag.extraction_runs import (
    ExtractionRun, select_sources, validate_chunk_parameters, verify_run,
)


class VersionedExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        pdf = self.root / 'corpus/raw/math/example.pdf'
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b'%PDF-fixture-for-hash-only')
        catalog = self.root / 'corpus/catalog/sources.yaml'
        catalog.parent.mkdir(parents=True)
        catalog.write_text(yaml.safe_dump({'sources': [{
            'id': 'math-test-001', 'title': 'Test source', 'lang': 'fr', 'domain': 'mathematiques',
            'level': 'L3', 'type': 'cours', 'topics': ['groupes'],
            'file': 'corpus/raw/math/example.pdf',
        }]}))
        self.sources = select_sources(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, run_id='test'):
        return ExtractionRun(self.root, self.sources, 'test', {}, __file__, run_id)

    def test_output_tamper_detected(self):
        run = self.make()
        (run.path / 'chunks.jsonl').write_text('{}\n')
        run.finish({'chunks': 1})
        self.assertEqual(verify_run(run.path)['status'], 'COMPLETE')
        (run.path / 'chunks.jsonl').write_text('modification')
        with self.assertRaises(ValueError):
            verify_run(run.path)

    def test_input_change_blocks_completion(self):
        run = self.make()
        (self.root / self.sources[0]['file']).write_bytes(b'changed')
        with self.assertRaises(RuntimeError):
            run.finish({})
        with self.assertRaises(ValueError):
            verify_run(run.path)

    def test_collision_and_incomplete_run(self):
        run = self.make()
        with self.assertRaises(FileExistsError):
            self.make()
        with self.assertRaises(ValueError):
            verify_run(run.path)
        second = self.make('test2')
        self.assertNotEqual(run.chunk_id('source', 1, 0), second.chunk_id('source', 1, 0))

    def test_unknown_source_and_parameters_fail(self):
        with self.assertRaises(ValueError):
            select_sources(self.root, ['inexistant'])
        for size, overlap in [(0, 0), (20, -1), (20, 20)]:
            with self.assertRaises(ValueError):
                validate_chunk_parameters(size, overlap)


if __name__ == '__main__':
    unittest.main()
