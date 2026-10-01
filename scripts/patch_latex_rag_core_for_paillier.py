from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f"[OK] Déjà appliqué : {label}")
        return text
    if old not in text:
        raise RuntimeError(
            f"Impossible d'appliquer le patch {label!r}: motif source introuvable."
        )
    print(f"[PATCH] {label}")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Patch minimal et idempotent de latex_rag_core.py pour "
            "préserver conjecture/property et exclure la bibliographie des embeddings."
        )
    )
    ap.add_argument(
        "--core",
        type=Path,
        default=Path("scripts/latex_rag_core.py"),
    )
    args = ap.parse_args()

    core = args.core
    if not core.exists():
        raise SystemExit(f"[ERREUR] Fichier introuvable : {core}")

    original = core.read_text(encoding="utf-8")
    text = original

    text = replace_once(
        text,
        '    "corollary": "corollary",\n',
        '    "corollary": "corollary",\n'
        '    "conjecture": "conjecture",\n'
        '    "property": "property",\n'
        '    "thebibliography": "bibliography",\n',
        "environnements conjecture/property/bibliography",
    )

    text = replace_once(
        text,
        '    "definition", "theorem", "proposition", "lemma", "corollary", "claim",\n',
        '    "definition", "theorem", "proposition", "lemma", "corollary", '
        '"conjecture", "property", "claim",\n',
        "conjecture/property comme statements",
    )

    text = replace_once(
        text,
        '    for unit in units:\n'
        '        latex = unit["latex"]\n',
        '    for unit in units:\n'
        '        if unit.get("unit_type") == "bibliography":\n'
        '            continue\n'
        '        latex = unit["latex"]\n',
        "bibliographie conservée dans units.jsonl mais non vectorisée",
    )

    if text == original:
        print("[OK] Aucun changement nécessaire.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = core.with_name(core.name + f".backup_{stamp}")
    shutil.copy2(core, backup)

    compile(text, str(core), "exec")
    core.write_text(text, encoding="utf-8")

    print(f"[OK] Backup : {backup}")
    print(f"[OK] Patch appliqué : {core}")
    print("[OK] Validation syntaxique Python réussie.")


if __name__ == "__main__":
    main()
