import argparse
from pathlib import Path
from livre_crypto_rag.extraction_runs import verify_run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Contrôle les empreintes des sorties ; pas la vérité mathématique.')
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify_run(args.run)
    print('Intégrité OK :', args.run)
    print('Comptages :', result['counts'])
