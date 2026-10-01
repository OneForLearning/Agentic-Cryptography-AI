from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


EXPECTED_TITLE = "The Paillier's Cryptosystem and Some Variants Revisited"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Installe additivement la source LaTeX Paillier dans le corpus "
            "sans effacer les cours déjà présents."
        )
    )
    ap.add_argument("archive", type=Path)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("corpus/raw/latex/expanded/crypto_paillier_revisited"),
    )
    args = ap.parse_args()

    archive = args.archive.expanduser().resolve()
    if not archive.exists():
        raise SystemExit(f"[ERREUR] Archive introuvable : {archive}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / "Paillier_latex.tex"

    with zipfile.ZipFile(archive) as zf:
        tex_files = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".tex")
            and not name.startswith("__MACOSX/")
        ]

        if len(tex_files) != 1:
            raise SystemExit(
                f"[ERREUR] Une seule source .tex attendue, trouvé : {tex_files}"
            )

        raw = zf.read(tex_files[0])

    text = raw.decode("utf-8", errors="strict")
    if EXPECTED_TITLE not in text:
        raise SystemExit(
            "[ERREUR] Le titre attendu n'a pas été trouvé. "
            "L'archive ne correspond peut-être pas à la source analysée."
        )

    target.write_bytes(raw)

    digest = hashlib.sha256(raw).hexdigest()

    print(f"[OK] Source installée : {target}")
    print(f"[OK] Taille : {len(raw)} octets")
    print(f"[OK] SHA256 : {digest}")
    print(f"[OK] Titre vérifié : {EXPECTED_TITLE}")
    print("[IMPORTANT] Aucun fichier existant du corpus n'a été supprimé.")


if __name__ == "__main__":
    main()
