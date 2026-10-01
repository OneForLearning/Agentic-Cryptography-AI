from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_COLLECTION = "ace_ai_crypto_latex_v1"


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Retrieve mathematical LaTeX context then ask a local Ollama LLM."
        )
    )

    ap.add_argument("question")
    ap.add_argument(
        "--llm",
        default="qwen3:14b",
    )
    ap.add_argument(
        "--retriever",
        type=Path,
        default=Path(__file__).with_name(
            "retrieve_latex.py"
        ),
    )
    ap.add_argument(
        "--processed",
        type=Path,
        default=Path("corpus/processed/latex"),
    )
    ap.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
    )
    ap.add_argument(
        "--model",
        default="bge-m3",
    )
    ap.add_argument(
        "--max-context-chars",
        type=int,
        default=30000,
    )

    args = ap.parse_args()

    cmd = [
        sys.executable,
        str(args.retriever),
        args.question,
        "--processed",
        str(args.processed),
        "--collection",
        args.collection,
        "--model",
        args.model,
        "--max-context-chars",
        str(args.max_context_chars),
        "--json",
    ]

    try:
        raw = subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        detail = (
            exc.stderr.strip()
            if exc.stderr
            else str(exc)
        )

        raise SystemExit(
            "Le retriever RAG a échoué.\n"
            + detail
        ) from exc

    chunks = json.loads(
        raw
    )

    if not chunks:
        raise SystemExit(
            "Aucun chunk récupéré : impossible de construire "
            "un contexte RAG fiable."
        )

    context = []

    for chunk in chunks:
        context.append(
            f"[CHUNK {chunk['chunk_id']}]\n"
            f"Course: {chunk.get('course_title')}\n"
            f"Section: {chunk.get('section_path_text')}\n"
            f"Type: {chunk.get('unit_type')}\n"
            f"Label: {chunk.get('label')}\n"
            f"Source file: {chunk.get('source_file')}\n"
            f"Retrieval reason: {chunk.get('_retrieval_reason')}\n\n"
            f"{chunk.get('latex', '')}"
        )

    joined = "\n\n".join(
        context
    )

    try:
        import ollama
    except ImportError as exc:
        raise SystemExit(
            "Dépendance manquante : uv add ollama"
        ) from exc

    system = """Tu réponds à partir du contexte RAG fourni.
Le LaTeX source est l'autorité documentaire.

Règles:
- N'invente pas un résultat absent du contexte.
- Pour un théorème, distingue l'énoncé de sa preuve.
- Si une preuve est incomplète ou qu'un document est signalé incomplet, dis-le.
- Conserve les hypothèses mathématiques.
- Cite les identifiants [CHUNK ...] qui supportent les affirmations importantes.
- Si plusieurs sources divergent, signale la divergence au lieu de les fusionner silencieusement."""

    prompt = (
        f"QUESTION:\n{args.question}\n\n"
        f"CONTEXTE RAG:\n{joined}"
    )

    response = ollama.chat(
        model=args.llm,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0.1,
        },
    )

    if isinstance(response, dict):
        content = response["message"]["content"]
    else:
        content = response.message.content

    print(content)


if __name__ == "__main__":
    main()
