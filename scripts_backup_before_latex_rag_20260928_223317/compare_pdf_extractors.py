import argparse
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from livre_crypto_rag.extraction_runs import ExtractionRun, select_sources, validate_chunk_parameters
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]

CATALOG_FILE = (
    ROOT
    / "corpus"
    / "catalog"
    / "sources.yaml"
)

STAGING_DIR = (
    ROOT
    / "corpus"
    / "processed"
    / "extraction_staging"
)

COMPARISON_FILE = (
    STAGING_DIR
    / "extraction_comparison.jsonl"
)

SELECTED_PAGES_FILE = (
    STAGING_DIR
    / "selected_pages.jsonl"
)

STAGED_CHUNKS_FILE = (
    STAGING_DIR
    / "chunks.jsonl"
)

TEXT_DUMP_DIR = (
    STAGING_DIR
    / "pages"
)

CHUNK_SIZE = 4000
OVERLAP = 400


# ============================================================
# BASIC IO
# ============================================================

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


# ============================================================
# TEXT SANITIZATION
# ============================================================

def sanitize_text(
    text: str,
) -> str:
    """
    Nettoyage minimal.

    Important :
    - ne compacte PAS toutes les lignes ;
    - ne supprime PAS les retours à la ligne ;
    - ne tente PAS de reconstruire les formules ;
    - ne fait PAS d'OCR.
    """

    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    cleaned = []

    for char in text:
        code = ord(char)

        if char in {
            "\n",
            "\t",
        }:
            cleaned.append(
                char
            )
            continue

        if code < 32:
            continue

        cleaned.append(
            char
        )

    text = "".join(
        cleaned
    )

    # Supprime uniquement les espaces en fin de ligne.
    lines = [
        line.rstrip()
        for line in text.splitlines()
    ]

    return "\n".join(
        lines
    ).strip()


# ============================================================
# EXTRACTION BACKENDS
# ============================================================

def extract_with_pypdf(
    pdf_path: Path,
) -> list[str]:
    reader = PdfReader(
        str(pdf_path)
    )

    pages = []

    for page in reader.pages:
        text = (
            page.extract_text()
            or ""
        )

        pages.append(
            sanitize_text(
                text
            )
        )

    return pages


def extract_with_pymupdf(
    pdf_path: Path,
) -> list[str] | None:
    try:
        import pymupdf
    except ImportError:
        return None

    document = pymupdf.open(
        str(pdf_path)
    )

    pages = []

    try:
        for page in document:
            # "text" préserve mieux la lecture naturelle ;
            # sort=True stabilise l'ordre des blocs.
            text = page.get_text(
                "text",
                sort=True,
            )

            pages.append(
                sanitize_text(
                    text
                )
            )
    finally:
        document.close()

    return pages


def extract_with_pdftotext_layout(
    pdf_path: Path,
) -> list[str] | None:
    executable = shutil.which(
        "pdftotext"
    )

    if executable is None:
        return None

    # On récupère d'abord le nombre de pages avec pypdf.
    page_count = len(
        PdfReader(
            str(pdf_path)
        ).pages
    )

    pages = []

    for page_number in range(
        1,
        page_count + 1,
    ):
        command = [
            executable,
            "-layout",
            "-enc",
            "UTF-8",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            str(pdf_path),
            "-",
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "pdftotext a échoué pour "
                f"{pdf_path}, page {page_number}:\n"
                + result.stderr.decode(
                    "utf-8",
                    errors="replace",
                )
            )

        text = result.stdout.decode(
            "utf-8",
            errors="replace",
        )

        # pdftotext ajoute généralement un form-feed.
        text = text.replace(
            "\f",
            ""
        )

        pages.append(
            sanitize_text(
                text
            )
        )

    return pages


# ============================================================
# QUALITY ANALYSIS
# ============================================================

def quality_analysis(
    text: str,
) -> dict:
    artifacts = []
    score = 0

    control_chars = sum(
        1
        for char in text
        if ord(char) < 32
        and char not in {
            "\n",
            "\t",
            "\r",
        }
    )

    if control_chars:
        artifacts.append(
            f"control_chars:{control_chars}"
        )
        score += min(
            6,
            control_chars,
        )

    replacement_chars = (
        text.count("�")
    )

    if replacement_chars:
        artifacts.append(
            f"replacement_chars:{replacement_chars}"
        )
        score += min(
            6,
            replacement_chars * 2,
        )

    spaced_mod = len(
        re.findall(
            r"\bm\s+o\s+d\b",
            text,
            flags=re.IGNORECASE,
        )
    )

    if spaced_mod:
        artifacts.append(
            f"broken_mod_tokens:{spaced_mod}"
        )
        score += (
            spaced_mod * 4
        )

    # Ex: "e f f o r t s" / "f i n i t e".
    spaced_sequences = len(
        re.findall(
            r"(?:\b[A-Za-z]\s+){3,}[A-Za-z]\b",
            text,
        )
    )

    if spaced_sequences:
        artifacts.append(
            f"spaced_letter_sequences:{spaced_sequences}"
        )
        score += min(
            8,
            spaced_sequences * 2,
        )

    lines = text.splitlines()

    nonempty_lines = [
        line
        for line in lines
        if line.strip()
    ]

    tiny_lines = [
        line
        for line in nonempty_lines
        if len(
            line.strip()
        ) <= 2
    ]

    tiny_ratio = (
        len(tiny_lines)
        / len(nonempty_lines)
        if nonempty_lines
        else 0.0
    )

    # On ne pénalise que si le ratio devient significatif.
    if (
        len(tiny_lines) >= 5
        and tiny_ratio >= 0.08
    ):
        artifacts.append(
            f"tiny_fragment_lines:{len(tiny_lines)}"
        )
        score += min(
            6,
            2
            + int(
                tiny_ratio * 10
            ),
        )

    # Signaux typiques du PDF actuel.
    broken_z_layout = len(
        re.findall(
            r"\bZ\s*\n\s*[npq](?:2)?\b",
            text,
            flags=re.IGNORECASE,
        )
    )

    if broken_z_layout:
        artifacts.append(
            f"broken_Z_layout:{broken_z_layout}"
        )
        score += (
            broken_z_layout * 3
        )

    suspicious_congruence = len(
        re.findall(
            r"(?:^|\n)\s*w\s*=\s*1\s*m?\s*o?\s*d?",
            text,
            flags=re.IGNORECASE,
        )
    )

    if suspicious_congruence:
        artifacts.append(
            f"broken_congruence_layout:{suspicious_congruence}"
        )
        score += (
            suspicious_congruence * 5
        )

    # "n2", "p2q" etc. peuvent être légitimes en texte extrait,
    # donc on ne les compte qu'avec des lignes mathématiques fragmentées.
    compact_math_fragments = len(
        re.findall(
            r"(?:^|\n)\s*[npq]\d+\s*(?:$|\n)",
            text,
            flags=re.IGNORECASE,
        )
    )

    if compact_math_fragments:
        artifacts.append(
            f"isolated_math_fragments:{compact_math_fragments}"
        )
        score += min(
            4,
            compact_math_fragments,
        )

    alpha_chars = sum(
        char.isalpha()
        for char in text
    )

    readable_chars = sum(
        char.isalnum()
        or char.isspace()
        or char in ".,;:!?()[]{}+-=*/<>_^%'\"\\|"
        for char in text
    )

    total_chars = len(
        text
    )

    readable_ratio = (
        readable_chars / total_chars
        if total_chars
        else 0.0
    )

    if total_chars < 100:
        artifacts.append(
            f"very_short_page:{total_chars}"
        )
        score += 5

    if (
        total_chars >= 100
        and readable_ratio < 0.92
    ):
        artifacts.append(
            f"low_readable_ratio:{readable_ratio:.3f}"
        )
        score += 3

    math_signal = bool(
        re.search(
            r"(?:mod|Z|λ|phi|lcm|gcd|≡|\^|n²|n2|p\s*[-−]\s*1)",
            text,
            flags=re.IGNORECASE,
        )
    )

    math_critical_artifact = any(
        artifact.startswith(
            (
                "broken_mod_tokens:",
                "broken_Z_layout:",
                "broken_congruence_layout:",
            )
        )
        for artifact in artifacts
    )

    if score <= 2:
        rating = "CLEAN"
    elif score <= 5:
        rating = "MINOR_ARTIFACTS"
    elif score <= 9:
        rating = "DEGRADED"
    else:
        rating = "SEVERELY_DEGRADED"

    text_reliable = (
        rating
        in {
            "CLEAN",
            "MINOR_ARTIFACTS",
        }
    )

    math_layout_heuristic_ok = (
        text_reliable
        and not math_critical_artifact
    )

    return {
        "artifact_score":
            score,
        "rating":
            rating,
        "text_reliable": False,
        "math_layout_heuristic_ok": math_layout_heuristic_ok,
        "text_layout_heuristic_ok": text_reliable,
        "math_reliable": False,
        "requires_human_source_check": True,
        "formula_fidelity": "NOT_REVIEWED",
        "mathematical_validity": "NOT_REVIEWED",
        "rag_admission": "BLOCKED",
        "quality_scope": "heuristic_only",
        "math_signal":
            math_signal,
        "math_critical_artifact":
            math_critical_artifact,
        "readable_ratio":
            round(
                readable_ratio,
                4,
            ),
        "text_length":
            total_chars,
        "alpha_chars":
            alpha_chars,
        "extraction_artifacts":
            artifacts,
    }


def backend_rank(
    quality: dict,
    backend_name: str,
) -> tuple:
    """
    Plus petit = meilleur.

    On préfère d'abord :
    1. math_layout_heuristic_ok (aucune validation mathématique)
    2. text_layout_heuristic_ok
    3. artifact_score faible
    4. texte assez riche
    5. pdftotext-layout en cas d'égalité quasi parfaite
       car il préserve souvent mieux la disposition.
    """

    backend_tie_break = {
        "pdftotext-layout": 0,
        "pymupdf": 1,
        "pypdf": 2,
    }.get(
        backend_name,
        9,
    )

    return (
        0
        if quality[
            "math_layout_heuristic_ok"
        ]
        else 1,
        0
        if quality[
            "text_layout_heuristic_ok"
        ]
        else 1,
        quality[
            "artifact_score"
        ],
        -min(
            quality[
                "text_length"
            ],
            10000,
        ),
        backend_tie_break,
    )


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap doit être < chunk_size."
        )

    chunks = []

    start = 0
    length = len(
        text
    )

    while start < length:
        target_end = min(
            start + chunk_size,
            length,
        )

        end = target_end

        # Cherche une frontière naturelle sans trop raccourcir.
        if target_end < length:
            search_start = max(
                start + int(
                    chunk_size * 0.65
                ),
                start,
            )

            candidates = [
                text.rfind(
                    "\n\n",
                    search_start,
                    target_end,
                ),
                text.rfind(
                    ". ",
                    search_start,
                    target_end,
                ),
                text.rfind(
                    "\n",
                    search_start,
                    target_end,
                ),
            ]

            natural = max(
                candidates
            )

            if natural > start:
                end = natural + 1

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(
                chunk
            )

        if end >= length:
            break

        next_start = max(
            0,
            end - overlap,
        )

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    global STAGING_DIR, COMPARISON_FILE, SELECTED_PAGES_FILE, STAGED_CHUNKS_FILE, TEXT_DUMP_DIR
    parser = argparse.ArgumentParser(
        description=(
            "Compare plusieurs extracteurs PDF et construit "
            "une extraction versionnée sous rag/runs sans modifier les sorties 6A.2."
        )
    )

    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Limiter à un source id. "
            "Option répétable."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
    )

    parser.add_argument("--run-id")
    args = parser.parse_args()

    validate_chunk_parameters(args.chunk_size, args.overlap)
    sources = select_sources(ROOT, args.source)
    run = ExtractionRun(ROOT, sources, "compare-three-extractors",
        {"chunk_size_characters": args.chunk_size, "overlap_characters": args.overlap,
         "selection": "heuristic-page-level-not-mathematical-certification"},
        __file__, args.run_id)
    STAGING_DIR = run.path
    COMPARISON_FILE = run.path / "extraction_comparison.jsonl"
    SELECTED_PAGES_FILE = run.path / "selected_pages.jsonl"
    STAGED_CHUNKS_FILE = run.path / "chunks.jsonl"
    TEXT_DUMP_DIR = run.path / "pages"

    comparison_records = []
    selected_page_records = []
    staged_chunks = []

    selected_backend_counts = Counter()
    selected_rating_counts = Counter()

    backend_availability = {
        "pypdf":
            True,
        "pymupdf":
            None,
        "pdftotext-layout":
            shutil.which(
                "pdftotext"
            )
            is not None,
    }

    try:
        import pymupdf  # noqa: F401
        backend_availability[
            "pymupdf"
        ] = True
    except ImportError:
        backend_availability[
            "pymupdf"
        ] = False

    print(
        "=== PDF Extraction Comparison ==="
    )

    print(
        "Backends:"
    )

    for name, available in (
        backend_availability.items()
    ):
        print(
            f"  {name:18s}: "
            f"{'OK' if available else 'ABSENT'}"
        )

    for source in sources:
        source_id = source.get(
            "id"
        )

        relative_file = source.get(
            "file"
        )

        if (
            not isinstance(
                source_id,
                str,
            )
            or not source_id
        ):
            raise RuntimeError(
                "Source sans id valide."
            )

        if (
            not isinstance(
                relative_file,
                str,
            )
            or not relative_file
        ):
            raise RuntimeError(
                f"{source_id}: file absent."
            )

        pdf_path = (
            ROOT
            / relative_file
        ).resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"{source_id}: PDF introuvable : {pdf_path}"
            )

        print()
        print(
            f"[SOURCE] {source_id}"
        )
        print(
            f"         {pdf_path}"
        )

        backend_pages = {}

        backend_pages[
            "pypdf"
        ] = extract_with_pypdf(
            pdf_path
        )

        pymupdf_pages = (
            extract_with_pymupdf(
                pdf_path
            )
        )

        if pymupdf_pages is not None:
            backend_pages[
                "pymupdf"
            ] = pymupdf_pages

        pdftotext_pages = (
            extract_with_pdftotext_layout(
                pdf_path
            )
        )

        if pdftotext_pages is not None:
            backend_pages[
                "pdftotext-layout"
            ] = pdftotext_pages

        page_counts = {
            name: len(
                pages
            )
            for name, pages
            in backend_pages.items()
        }

        if len(
            set(
                page_counts.values()
            )
        ) != 1:
            raise RuntimeError(
                f"{source_id}: nombre de pages incohérent "
                f"entre extracteurs : {page_counts}"
            )

        page_count = next(
            iter(
                page_counts.values()
            )
        )

        source_dump_dir = (
            TEXT_DUMP_DIR
            / source_id
        )

        source_dump_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for page_index in range(
            page_count
        ):
            page_number = (
                page_index + 1
            )

            candidates = {}

            for backend_name, pages in (
                backend_pages.items()
            ):
                text = pages[
                    page_index
                ]

                quality = (
                    quality_analysis(
                        text
                    )
                )

                candidates[
                    backend_name
                ] = {
                    "quality":
                        quality,
                    "text":
                        text,
                }

                dump_file = (
                    source_dump_dir
                    / (
                        f"page_{page_number:04d}"
                        f".{backend_name}.txt"
                    )
                )

                dump_file.write_text(
                    text,
                    encoding="utf-8",
                )

            selected_backend = min(
                candidates,
                key=lambda name:
                    backend_rank(
                        candidates[
                            name
                        ][
                            "quality"
                        ],
                        name,
                    ),
            )

            selected = candidates[
                selected_backend
            ]

            selected_backend_counts[
                selected_backend
            ] += 1

            selected_rating_counts[
                selected[
                    "quality"
                ][
                    "rating"
                ]
            ] += 1

            comparison_records.append(
                {
                    "source_id":
                        source_id,
                    "page":
                        page_number,
                    "selected_backend":
                        selected_backend,
                    "selected_quality":
                        selected[
                            "quality"
                        ],
                    "backends": {
                        name:
                            data[
                                "quality"
                            ]
                        for name, data
                        in candidates.items()
                    },
                }
            )

            selected_page_records.append(
                {
                    "source_id":
                        source_id,
                    "page":
                        page_number,
                    "extractor":
                        selected_backend,
                    "source_quality":
                        selected[
                            "quality"
                        ],
                    "text":
                        selected[
                            "text"
                        ],
                }
            )

            chunks = chunk_text(
                selected[
                    "text"
                ],
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )

            for chunk_index, chunk in enumerate(
                chunks
            ):
                staged_chunks.append(
                    {
                        "chunk_id":
                            run.chunk_id(source_id, page_number, chunk_index),
                        "run_id": run.run_id,
                        "source_sha256": run.source_hash(source),
                        "validation_status": "NOT_REVIEWED",
                        "rag_admission": "BLOCKED",
                        "quality_scope": "page_heuristic",
                        "source_id":
                            source_id,
                        "domain":
                            source.get(
                                "domain"
                            ),
                        "level":
                            source.get(
                                "level"
                            ),
                        "topics":
                            source.get(
                                "topics",
                                [],
                            ),
                        "page":
                            page_number,
                        "chunk_index":
                            chunk_index,
                        "extractor":
                            selected_backend,
                        "source_quality":
                            selected[
                                "quality"
                            ],
                        "text":
                            chunk,
                    }
                )

    write_jsonl(
        COMPARISON_FILE,
        comparison_records,
    )

    write_jsonl(
        SELECTED_PAGES_FILE,
        selected_page_records,
    )

    write_jsonl(
        STAGED_CHUNKS_FILE,
        staged_chunks,
    )

    run.finish({"sources": len(sources), "pages": len(selected_page_records),
                "chunks": len(staged_chunks), "backends": dict(selected_backend_counts),
                "page_quality_heuristic": dict(selected_rating_counts)})
    print("Extraction COMPLETE :", run.path)

    print()
    print(
        "=== Résumé ==="
    )

    print(
        f"Pages comparées : "
        f"{len(comparison_records)}"
    )

    print(
        "Backends sélectionnés :",
        dict(
            sorted(
                selected_backend_counts.items()
            )
        ),
    )

    print(
        "Qualité sélectionnée :",
        dict(
            sorted(
                selected_rating_counts.items()
            )
        ),
    )

    print()
    print(
        f"[OK] Comparaison : "
        f"{COMPARISON_FILE}"
    )
    print(
        f"[OK] Pages retenues : "
        f"{SELECTED_PAGES_FILE}"
    )
    print(
        f"[OK] Chunks staging : "
        f"{STAGED_CHUNKS_FILE}"
    )
    print(
        f"[OK] Dumps texte : "
        f"{TEXT_DUMP_DIR}"
    )

    print()
    print(
        "IMPORTANT : le fichier de production "
        "corpus/processed/chunks.jsonl "
        "n'a PAS été modifié."
    )


if __name__ == "__main__":
    main()
