import importlib.util
from pathlib import Path
import unittest
from livre_crypto_rag.math_review import build_units, assert_coverage

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparator', ROOT / 'scripts/compare_pdf_extractors.py')
comparator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparator)


class MathReviewTests(unittest.TestCase):
    def pages(self):
        return [
            {'source_id': 'test', 'page': 1, 'text': 'Lemma 3. A statement.\n'},
            {'source_id': 'test', 'page': 2, 'text': '226     Pascal Paillier\nProof. First half.\n'},
            {'source_id': 'test', 'page': 3, 'text': 'Second half.\nTheorem 9. Another result.\nLemma 10. Helper.\nProof (of Lemma 10). Helper proof.\nProof (of Theorem 9). Main proof.\n'},
        ]

    def test_cross_page_coverage_and_proofs(self):
        pages = self.pages()
        units = build_units(pages, 'run')
        assert_coverage(pages, units)
        proof = next(u for u in units if u['heading'].startswith('Proof.'))
        lemma = next(u for u in units if u['kind'] == 'lemma' and u['number'] == '3')
        self.assertEqual(proof['proof_of_candidate'], lemma['unit_id'])
        self.assertEqual(proof['pages'], [2, 3])
        main = next(u for u in units if u['heading'].startswith('Proof (of Theorem'))
        target = next(u for u in units if u['kind'] == 'theorem')
        self.assertEqual(main['proof_of_candidate'], target['unit_id'])
        self.assertTrue(all(u['rag_admission'] == 'BLOCKED' for u in units))
        for u in units:
            reconstructed = ''.join(next(p['text'] for p in pages if p['source_id'] == u['source_id'] and p['page'] == s['page'])[s['start']:s['end']] for s in u['segments'])
            self.assertEqual(reconstructed, u['text'])

    def test_internal_citation_is_not_proof_target(self):
        pages = [{'source_id':'a','page':1,'text':'Theorem 11. Earlier result.\nTheorem 16. Target.\nProof. By Theorem 11 this follows.\n'}]
        units = build_units(pages, 'run')
        self.assertEqual(units[-1]['proof_of_candidate'], units[1]['unit_id'])
        self.assertEqual(units[-1]['link_basis'], 'preceding_result')

    def test_no_cross_source_link(self):
        pages = [{'source_id':'a','page':1,'text':'Theorem 1. A\n'},
                 {'source_id':'b','page':1,'text':'Proof. Unknown\n'}]
        units = build_units(pages, 'run')
        self.assertIsNone(units[-1]['proof_of_candidate'])

    def test_ambiguous_explicit_reference_is_not_linked(self):
        pages = [{'source_id':'a','page':1,'text':'Lemma 1. A\nLemma 1. B\nProof (of Lemma 1). C\n'}]
        self.assertIsNone(build_units(pages, 'run')[-1]['proof_of_candidate'])

    def test_coverage_detects_missing_text(self):
        pages = self.pages()
        units = build_units(pages, 'run')
        units[0]['segments'][0]['end'] -= 1
        with self.assertRaises(ValueError):
            assert_coverage(pages, units)

    def test_clean_is_never_math_validation(self):
        # Function name found from the actual comparator, not a test-side substitute.
        q = comparator.quality_analysis('This is a long clean sentence about groups and their definitions. ' * 20)
        self.assertFalse(q['math_reliable'])
        self.assertFalse(q['text_reliable'])
        self.assertTrue(q['requires_human_source_check'])
        self.assertEqual(q['formula_fidelity'], 'NOT_REVIEWED')
        self.assertEqual(q['rag_admission'], 'BLOCKED')
        comparator.backend_rank(q, 'pypdf')


if __name__ == '__main__':
    unittest.main()
