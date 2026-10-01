from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path

NS = uuid.UUID("d6f1c6e5-2b93-4a5a-a3cd-9bc853f6200e")
BOOK_TITLE = "Cryptography: An Introduction (3rd Edition)"
AUTHOR = "Nigel Smart"
GENERIC_EMPTY = {"Chapter Summary", "Further Reading"}


def uid(prefix: str, *parts: object) -> str:
    u = uuid.uuid5(NS, "|".join(map(str, parts)))
    return f"{prefix}-{u.hex[:24]}"


def qid(*parts: object) -> str:
    return str(uuid.uuid5(NS, "qdrant|" + "|".join(map(str, parts))))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def strip_comment_line(line: str) -> str:
    out = []
    for i, c in enumerate(line):
        if c == "%":
            backslashes = 0
            j = i - 1
            while j >= 0 and line[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                break
        out.append(c)
    return "".join(out)


def parse_braced(s: str, start: int):
    if start >= len(s) or s[start] != "{":
        return None, None
    depth = 0
    i = start
    while i < len(s):
        c = s[i]
        if c in "{}":
            backslashes = 0
            j = i - 1
            while j >= 0 and s[j] == "\\":
                backslashes += 1
                j -= 1
            escaped = backslashes % 2 == 1
            if not escaped:
                if c == "{":
                    depth += 1
                else:
                    depth -= 1
                    if depth == 0:
                        return s[start + 1 : i], i + 1
        i += 1
    return None, None


def extract_heading(line: str):
    clean = strip_comment_line(line).strip()
    m = re.match(
        r"\\(part|chapter|section|subsection|subsubsection)(\*)?\s*\{",
        clean,
    )
    if not m:
        return None
    kind = m.group(1) + ("*" if m.group(2) else "")
    brace = clean.find("{", m.start())
    title, end = parse_braced(clean, brace)
    if title is None:
        return None
    rest = clean[end:].strip()
    lm = re.search(r"\\label\{([^{}]+)\}", rest)
    label = lm.group(1) if lm else None
    return kind, title, label, clean


def plain_latex(s: str) -> str:
    out = s
    replacements = {
        r"\varphi": "φ",
        r"\phi": "φ",
        r"\mathcal{NP}": "NP",
        r"\mathcal{P}": "P",
        r"\mathbb{Z}": "Z",
        r"\mathbb{F}": "F",
        r"\mathbb{Q}": "Q",
        r"\times": "×",
        r"\ge": "≥",
        r"\le": "≤",
    }
    for a, b in replacements.items():
        out = out.replace(a, b)
    out = out.replace("$", "")
    out = re.sub(r"\\text\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\mathcal\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\mathbb\{([^{}]*)\}", r"\1", out)
    out = out.replace("{", "").replace("}", "")
    out = re.sub(r"\\([A-Za-z]+)", r"\1", out)
    return re.sub(r"\s+", " ", out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    archive_bytes = args.archive.read_bytes()
    source_sha = hashlib.sha256(archive_bytes).hexdigest()

    with zipfile.ZipFile(args.archive) as z:
        names = [n for n in z.namelist() if n.endswith(".tex")]
        texts = {
            n: z.read(n).decode("utf-8", errors="replace")
            for n in names
        }

    if "smart/main.tex" not in texts:
        raise RuntimeError("smart/main.tex absent de l'archive")

    main_text = texts["smart/main.tex"]

    parts = []
    input_part = {}
    current_part = None
    for line_no, line in enumerate(main_text.splitlines(), 1):
        heading = extract_heading(line)
        if heading and heading[0] == "part":
            current_part = heading[1]
            parts.append(
                (heading[1], heading[2], line_no, heading[3])
            )

        m = re.search(
            r"\\input\{([^}]+)\}",
            strip_comment_line(line),
        )
        if m:
            path = m.group(1)
            if not path.endswith(".tex"):
                path += ".tex"
            input_part["smart/" + path] = current_part

    documents = []
    document_ids = {}
    for source_file in sorted(texts):
        document_id = uid(
            "doc", source_sha, source_file
        )
        document_ids[source_file] = document_id
        documents.append(
            {
                "document_id": document_id,
                "source_file": source_file,
                "source_archive": args.archive.name,
                "source_sha256": source_sha,
                "title": BOOK_TITLE,
                "author": AUTHOR,
                "content_kind": "outline",
                "knowledge_status": "skeleton_only",
                "exclude_from_answer_generation": True,
            }
        )

    units = []
    chunks = []
    relations = []
    macros = []
    declared_environments = []
    excluded_headings = []
    sibling_groups = defaultdict(list)
    units_by_id = {}

    def add_unit(
        *,
        source_file: str,
        line_no: int,
        unit_type: str,
        title_latex: str,
        raw_latex: str,
        label: str | None,
        parent_id: str | None,
        section_path: list[str],
        part_title: str | None = None,
        is_appendix: bool = False,
    ):
        document_id = document_ids[source_file]
        title_plain = plain_latex(title_latex)
        unit_id = uid(
            "unit",
            source_sha,
            source_file,
            line_no,
            unit_type,
            title_latex,
        )

        unit = {
            "unit_id": unit_id,
            "document_id": document_id,
            "unit_type": unit_type,
            "semantic_role": "topic_router",
            "content_kind": "outline",
            "knowledge_status": "title_only",
            "title_latex": title_latex,
            "title_plain": title_plain,
            "section_path": section_path,
            "source_file": source_file,
            "source_line_start": line_no,
            "source_line_end": line_no,
            "label": label,
            "parent_unit_id": parent_id,
            "exclude_from_answer_generation": True,
            "source_latex": raw_latex,
            "part_title": part_title,
            "is_appendix": is_appendix,
        }
        units.append(unit)
        units_by_id[unit_id] = unit

        if parent_id:
            relations.append(
                {
                    "relation_id": uid(
                        "rel",
                        "parent_of",
                        parent_id,
                        unit_id,
                    ),
                    "type": "parent_of",
                    "from_unit_id": parent_id,
                    "to_unit_id": unit_id,
                    "resolved": True,
                    "relation_strength": "structural",
                }
            )

        sibling_groups[parent_id].append(unit_id)

        role = unit_type.replace("outline_", "").capitalize()
        hierarchy = " > ".join(section_path)
        embedding_text = (
            f"Book: {BOOK_TITLE}\n"
            f"Author: {AUTHOR}\n"
            f"Navigation role: {role}\n"
            f"Topic path: {hierarchy}\n"
            f"Topic: {title_plain}\n"
            "Knowledge status: outline only. This source provides the "
            "topic heading but not its mathematical or cryptographic "
            "definition, theorem statement, formula, proof, hypothesis, "
            "or conclusion."
        )

        chunks.append(
            {
                "chunk_id": uid("chunk", unit_id),
                "qdrant_id": qid(unit_id),
                "unit_id": unit_id,
                "document_id": document_id,
                "unit_type": unit_type,
                "semantic_role": "topic_router",
                "content_kind": "outline",
                "retrieval_role": "topic_router",
                "knowledge_status": "title_only",
                "exclude_from_answer_generation": True,
                "book_title": BOOK_TITLE,
                "author": AUTHOR,
                "source_file": source_file,
                "source_line_start": line_no,
                "source_line_end": line_no,
                "title": title_plain,
                "title_latex": title_latex,
                "section_path": section_path,
                "label": label,
                "parent_unit_id": parent_id,
                "part_title": part_title,
                "is_appendix": is_appendix,
                "embedding_text": embedding_text,
                "latex": raw_latex,
                "content_hash": sha256_text(raw_latex),
            }
        )

        return unit_id

    part_unit_by_title = {}
    for title, label, line_no, raw in parts:
        title_plain = plain_latex(title)
        unit_id = add_unit(
            source_file="smart/main.tex",
            line_no=line_no,
            unit_type="outline_part",
            title_latex=title,
            raw_latex=raw,
            label=label,
            parent_id=None,
            section_path=[title_plain],
        )
        part_unit_by_title[title] = unit_id

    for source_file in sorted(texts):
        if source_file == "smart/main.tex":
            continue

        part_title_raw = input_part.get(source_file)
        part_id = part_unit_by_title.get(part_title_raw)
        part_title_plain = (
            plain_latex(part_title_raw)
            if part_title_raw
            else None
        )

        current_chapter = None
        current_section = None
        current_subsection = None

        for line_no, line in enumerate(
            texts[source_file].splitlines(), 1
        ):
            heading = extract_heading(line)
            if not heading:
                continue

            kind, title, label, raw = heading
            title_plain = plain_latex(title)

            if (
                kind == "section*"
                and title_plain in GENERIC_EMPTY
            ):
                excluded_headings.append(
                    {
                        "source_file": source_file,
                        "source_line": line_no,
                        "kind": kind,
                        "title": title_plain,
                        "reason": (
                            "generic_empty_navigation_heading"
                        ),
                    }
                )
                continue

            if kind.startswith("chapter"):
                unit_type = "outline_chapter"
                parent_id = part_id
                path = (
                    [part_title_plain]
                    if part_title_plain
                    else []
                ) + [title_plain]
            elif kind.startswith("section"):
                unit_type = "outline_section"
                parent_id = current_chapter or part_id
                parent_path = (
                    units_by_id.get(parent_id, {})
                    .get("section_path", [])
                )
                path = parent_path + [title_plain]
            elif kind.startswith("subsection"):
                unit_type = "outline_subsection"
                parent_id = (
                    current_section
                    or current_chapter
                    or part_id
                )
                parent_path = (
                    units_by_id.get(parent_id, {})
                    .get("section_path", [])
                )
                path = parent_path + [title_plain]
            else:
                unit_type = "outline_subsubsection"
                parent_id = (
                    current_subsection
                    or current_section
                    or current_chapter
                    or part_id
                )
                parent_path = (
                    units_by_id.get(parent_id, {})
                    .get("section_path", [])
                )
                path = parent_path + [title_plain]

            unit_id = add_unit(
                source_file=source_file,
                line_no=line_no,
                unit_type=unit_type,
                title_latex=title,
                raw_latex=raw,
                label=label,
                parent_id=parent_id,
                section_path=path,
                part_title=part_title_plain,
                is_appendix=(
                    source_file.endswith("annexeA.tex")
                    and unit_type == "outline_chapter"
                ),
            )

            if unit_type == "outline_chapter":
                current_chapter = unit_id
                current_section = None
                current_subsection = None
            elif unit_type == "outline_section":
                current_section = unit_id
                current_subsection = None
            elif unit_type == "outline_subsection":
                current_subsection = unit_id

    for _, ids in sibling_groups.items():
        for first, second in zip(ids, ids[1:]):
            relations.append(
                {
                    "relation_id": uid(
                        "rel",
                        "next_sibling",
                        first,
                        second,
                    ),
                    "type": "next_sibling",
                    "from_unit_id": first,
                    "to_unit_id": second,
                    "resolved": True,
                    "relation_strength": "structural",
                }
            )

    for line_no, line in enumerate(
        main_text.splitlines(), 1
    ):
        clean = strip_comment_line(line).strip()

        for m in re.finditer(
            r"\\newcommand\{\\([A-Za-z]+)\}",
            clean,
        ):
            name = "\\" + m.group(1)
            macros.append(
                {
                    "macro_id": uid(
                        "macro",
                        source_sha,
                        "smart/main.tex",
                        line_no,
                        name,
                    ),
                    "document_id": document_ids[
                        "smart/main.tex"
                    ],
                    "name": name,
                    "definition_latex": clean,
                    "source_file": "smart/main.tex",
                    "source_line": line_no,
                }
            )

        m = re.match(
            r"\\newtheorem(\*)?\{([^}]+)\}",
            clean,
        )
        if m:
            declared_environments.append(
                {
                    "environment": m.group(2),
                    "starred": bool(m.group(1)),
                    "source_file": "smart/main.tex",
                    "source_line": line_no,
                    "declaration_latex": clean,
                }
            )

    clean_all = "\n".join(
        strip_comment_line(line)
        for text in texts.values()
        for line in text.splitlines()
    )

    formal_names = [
        d["environment"]
        for d in declared_environments
    ] + ["proof"]

    formal_usage = {
        env: len(
            re.findall(
                r"\\begin\{" + re.escape(env) + r"\}",
                clean_all,
            )
        )
        for env in sorted(set(formal_names))
    }

    display_envs = [
        "equation",
        "equation*",
        "align",
        "align*",
        "gather",
        "gather*",
        "multline",
        "multline*",
        "displaymath",
    ]
    display_count = sum(
        len(
            re.findall(
                r"\\begin\{" + re.escape(env) + r"\}",
                clean_all,
            )
        )
        for env in display_envs
    )
    display_count += len(
        re.findall(r"\\\[", clean_all)
    )

    ref_count = len(
        re.findall(
            r"\\(?:ref|eqref|pageref)\{",
            clean_all,
        )
    )
    label_count = len(
        re.findall(r"\\label\{", clean_all)
    )
    placeholder_comments = sum(
        text.count(">>>")
        for text in texts.values()
    )

    stats = {
        "source_archive": args.archive.name,
        "source_sha256": source_sha,
        "archive_tex_files": len(texts),
        "documents": len(documents),
        "outline_chunks": len(chunks),
        "knowledge_chunks": 0,
        "units": len(units),
        "relations": len(relations),
        "relation_types": dict(
            __import__("collections").Counter(
                r["type"] for r in relations
            )
        ),
        "unit_types": dict(
            __import__("collections").Counter(
                u["unit_type"] for u in units
            )
        ),
        "parts": sum(
            u["unit_type"] == "outline_part"
            for u in units
        ),
        "chapters": sum(
            u["unit_type"] == "outline_chapter"
            for u in units
        ),
        "sections": sum(
            u["unit_type"] == "outline_section"
            for u in units
        ),
        "subsections": sum(
            u["unit_type"] == "outline_subsection"
            for u in units
        ),
        "excluded_generic_empty_headings": len(
            excluded_headings
        ),
        "placeholder_comments": placeholder_comments,
        "macro_declarations": len(macros),
        "declared_environments": len(
            declared_environments
        ),
        "formal_environment_usage": formal_usage,
        "content_display_math_blocks": display_count,
        "content_refs": ref_count,
        "labels": label_count,
        "max_embedding_chars": max(
            len(c["embedding_text"])
            for c in chunks
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.out / "documents.jsonl",
        documents,
    )
    write_jsonl(
        args.out / "macros.jsonl",
        macros,
    )
    write_jsonl(
        args.out / "units.jsonl",
        units,
    )
    write_jsonl(
        args.out / "relations.jsonl",
        relations,
    )
    write_jsonl(
        args.out / "chunks.jsonl",
        chunks,
    )
    write_jsonl(
        args.out / "declared_environments.jsonl",
        declared_environments,
    )
    write_jsonl(
        args.out / "excluded_headings.jsonl",
        excluded_headings,
    )
    (args.out / "stats.json").write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
