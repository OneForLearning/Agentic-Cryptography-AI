from __future__ import annotations
import argparse, shutil, zipfile
from pathlib import Path

IGNORE_PREFIXES = ('__MACOSX/',)
IGNORE_NAMES = {'.DS_Store'}

def safe_extract(zp: Path, dest: Path) -> None:
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            name = info.filename
            if name.startswith(IGNORE_PREFIXES) or Path(name).name in IGNORE_NAMES:
                continue
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f'Unsafe zip path: {name}')
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, target.open('wb') as out:
                    shutil.copyfileobj(src, out)

def main():
    ap=argparse.ArgumentParser(description='Recursively unpack LaTeX course ZIPs for RAG ingestion.')
    ap.add_argument('archive', type=Path)
    ap.add_argument('--out', type=Path, default=Path('corpus/raw/latex/expanded'))
    ap.add_argument('--clean', action='store_true')
    args=ap.parse_args()
    if args.clean and args.out.exists(): shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    staging=args.out/'_outer'
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True)
    safe_extract(args.archive, staging)
    nested=[]
    for zp in sorted(staging.rglob('*.zip')):
        if '__MACOSX' in zp.parts: continue
        nested.append(zp)
    for zp in nested:
        name=zp.stem
        target=args.out/name
        target.mkdir(parents=True, exist_ok=True)
        safe_extract(zp,target)
    # Also copy any top-level .tex not inside nested zip archives.
    for tex in staging.rglob('*.tex'):
        rel=tex.relative_to(staging)
        target=args.out/'loose'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(tex,target)
    shutil.rmtree(staging)
    print(f'[OK] Corpus préparé : {args.out.resolve()}')
    print(f'TeX files: {sum(1 for _ in args.out.rglob("*.tex"))}')

if __name__=='__main__': main()
