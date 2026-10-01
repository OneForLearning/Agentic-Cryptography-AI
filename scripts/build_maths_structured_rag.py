
from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from bisect import bisect_right
from collections import Counter
from pathlib import Path

DOC_NAMESPACE = uuid.UUID("ea5a2dd2-1e45-4bfb-92d9-a0c5787f51fd")

FORMAL_MARKER_RE = re.compile(
    r"%\s*RAG_UNIT_BEGIN\s+(?P<attrs>[^\n]+)\n"
    r"(?P<body>.*?)"
    r"%\s*RAG_UNIT_END\s+type=(?P<endtype>[^\s\n]+)",
    re.S,
)
ATTR_RE = re.compile(r"(\w+)=([^\s]+)")
HEADING_CMD_RE = re.compile(
    r"\\(part|chapter|section|subsection|subsubsection)(\*)?\s*\{"
)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|pageref|autoref|cref|Cref)\{([^}]+)\}")
BEGIN_ENV_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|empheq)\}"
)
DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.S)

RAG_COMMENT_RE = re.compile(r"^\s*%\s*RAG_[^\n]*$", re.M)
COMMENT_LINE_RE = re.compile(r"^\s*%.*$", re.M)

LEVELS = {
    "part": 0,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "subsubsection": 4,
}


def sha(text: str, n: int = 20) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    return raw.decode("utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def parse_balanced(s: str, brace_start: int) -> tuple[str, int]:
    if brace_start >= len(s) or s[brace_start] != "{":
        raise ValueError("parse_balanced: expected {")
    depth = 0
    i = brace_start
    while i < len(s):
        ch = s[i]
        escaped = i > 0 and s[i - 1] == "\\"
        if ch == "{" and not escaped:
            depth += 1
        elif ch == "}" and not escaped:
            depth -= 1
            if depth == 0:
                return s[brace_start + 1 : i], i + 1
        i += 1
    return s[brace_start + 1 :], len(s)


def visible_title(s: str) -> str:
    x = s
    x = re.sub(r"\\textsc\{([^{}]*)\}", r"\1", x)
    x = re.sub(r"\\(?:textbf|emph|mathrm|mathbf)\{([^{}]*)\}", r"\1", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def scan_headings(text: str) -> list[dict]:
    raw = []
    for ordinal, m in enumerate(HEADING_CMD_RE.finditer(text), 1):
        title, end = parse_balanced(text, m.end() - 1)
        raw.append(
            {
                "pos": m.start(),
                "end": end,
                "kind": m.group(1),
                "starred": bool(m.group(2)),
                "title_latex": title,
                "title": visible_title(title),
                "ordinal": ordinal,
            }
        )

    state: list[dict | None] = [None] * 5
    sections = []

    for i, h in enumerate(raw):
        level = LEVELS[h["kind"]]
        state[level] = h
        for j in range(level + 1, 5):
            state[j] = None

        path = [
            {
                "kind": item["kind"],
                "title": item["title"],
                "title_latex": item["title_latex"],
            }
            for item in state
            if item is not None
        ]

        section_id = "sec-maths-" + sha(
            "|".join(
                f"{p['kind']}:{p['title_latex']}"
                for p in path
            )
            + f"|{h['ordinal']}",
            24,
        )

        next_pos = raw[i + 1]["pos"] if i + 1 < len(raw) else len(text)
        label_window = text[h["end"] : min(next_pos, h["end"] + 400)]
        lm = LABEL_RE.search(label_window)
        label = lm.group(1) if lm else None

        rec = {
            "section_id": section_id,
            "kind": h["kind"],
            "level": level,
            "title": h["title"],
            "title_latex": h["title_latex"],
            "section_path": path,
            "section_path_text": " > ".join(p["title"] for p in path),
            "label": label,
            "source_pos": h["pos"],
        }
        sections.append(rec)
        h["section_id"] = section_id
        h["section_path"] = path
        h["section_path_text"] = rec["section_path_text"]

    return sections


def section_at(pos: int, sections: list[dict]) -> dict | None:
    positions = [s["source_pos"] for s in sections]
    idx = bisect_right(positions, pos) - 1
    return sections[idx] if idx >= 0 else None


def extract_env_title(body: str) -> str | None:
    m = re.search(
        r"\\begin\{(?:theorem|itheorem|prop|lemme|dfn|methode|exemple)\}"
        r"(?:\[([^\]]+)\])?",
        body,
    )
    if m and m.group(1):
        return visible_title(m.group(1))
    return None


def strip_rag_comments(s: str) -> str:
    return RAG_COMMENT_RE.sub("", s)


def strip_comments_for_embedding(s: str) -> str:
    # Source latex remains untouched in `latex`; only embedding text drops comment-only lines.
    return COMMENT_LINE_RE.sub("", s)


def strip_structural_commands(s: str) -> str:
    # Section commands and their immediately following section labels are metadata.
    out = []
    i = 0
    while i < len(s):
        m = HEADING_CMD_RE.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        _, end = parse_balanced(s, m.end() - 1)
        i = end
        lm = re.match(r"\s*\\label\{[^}]+\}", s[i:])
        if lm:
            i += lm.end()
    return "".join(out)


def normalize_gap(s: str) -> str:
    s = strip_rag_comments(s)
    s = strip_structural_commands(s)
    s = strip_comments_for_embedding(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def classify_gap(body: str) -> str:
    lower = body.lower().lstrip()
    if (
        lower.startswith(r"\paragraph{attention.")
        or lower.startswith(r"\attention{")
    ):
        return "warning"
    if (
        lower.startswith(r"\textbf{remarque")
        or lower.startswith(r"\emph{remarque")
        or lower.startswith("remarque")
    ):
        return "remark"
    if (
        "hypothèse" in lower[:500]
        and len(body) <= 600
    ):
        return "hypothesis_note"
    return "prose"


def brace_balance(s: str) -> tuple[int, int]:
    bal = 0
    min_bal = 0
    i = 0
    while i < len(s):
        if s[i] in "{}" and not (i > 0 and s[i - 1] == "\\"):
            bal += 1 if s[i] == "{" else -1
            min_bal = min(min_bal, bal)
        i += 1
    return bal, min_bal


def next_env_end(s: str, start: int, env: str) -> int | None:
    # The targeted display environments in this corpus are not recursively nested
    # with the same name in the same formula block.
    end_marker = rf"\end{{{env}}}"
    j = s.find(end_marker, start)
    return None if j < 0 else j + len(end_marker)


def extract_formulas(unit_id: str, body: str) -> list[dict]:
    formulas = []
    occupied = []

    for m in BEGIN_ENV_RE.finditer(body):
        env = m.group(1)
        end = next_env_end(body, m.end(), env)
        if end is None:
            continue
        # Avoid double-counting a begin marker that lies inside an already captured block.
        if any(a <= m.start() < b for a, b in occupied):
            continue
        latex = body[m.start():end]
        idx = len(formulas) + 1
        fid = "formula-maths-" + sha(f"{unit_id}|{idx}|{latex}", 24)
        labels = LABEL_RE.findall(latex)
        formulas.append(
            {
                "formula_id": fid,
                "unit_id": unit_id,
                "formula_index": idx,
                "environment": env,
                "latex": latex,
                "labels": labels,
                "refs": REF_RE.findall(latex),
                "content_hash": hashlib.sha256(
                    latex.encode("utf-8")
                ).hexdigest(),
            }
        )
        occupied.append((m.start(), end))

    for m in DISPLAY_RE.finditer(body):
        if any(a <= m.start() < b for a, b in occupied):
            continue
        latex = m.group(0)
        idx = len(formulas) + 1
        fid = "formula-maths-" + sha(f"{unit_id}|{idx}|{latex}", 24)
        formulas.append(
            {
                "formula_id": fid,
                "unit_id": unit_id,
                "formula_index": idx,
                "environment": "displaymath",
                "latex": latex,
                "labels": LABEL_RE.findall(latex),
                "refs": REF_RE.findall(latex),
                "content_hash": hashlib.sha256(
                    latex.encode("utf-8")
                ).hexdigest(),
            }
        )

    return formulas


def make_relation(
    rel_type: str,
    from_unit_id: str,
    *,
    to_unit_id: str | None = None,
    target_label: str | None = None,
    target_section_id: str | None = None,
    evidence: str,
    confidence: str = "explicit",
) -> dict:
    payload = {
        "type": rel_type,
        "from_unit_id": from_unit_id,
        "to_unit_id": to_unit_id,
        "target_label": target_label,
        "target_section_id": target_section_id,
        "evidence": evidence,
        "confidence": confidence,
    }
    rid = "rel-" + sha(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        32,
    )
    return {"relation_id": rid, **payload}


def plainish_latex(s: str) -> str:
    x = strip_comments_for_embedding(s)
    x = re.sub(r"%.*", "", x)
    x = re.sub(r"\\label\{[^}]+\}", "", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()


def build(source: Path, out: Path) -> dict:
    text = read_text(source)
    raw_bytes = source.read_bytes()
    source_sha = hashlib.sha256(raw_bytes).hexdigest()
    doc_id = "doc-maths-" + source_sha[:20]

    sections = scan_headings(text)

    formal_matches = list(FORMAL_MARKER_RE.finditer(text))
    source_id_to_unit: dict[str, str] = {}
    formal_units: list[dict] = []

    for ordinal, m in enumerate(formal_matches, 1):
        attrs = dict(ATTR_RE.findall(m.group("attrs")))
        source_type = attrs.get("type", m.group("endtype"))
        source_id = attrs.get("id")
        depends_on = attrs.get("depends_on")

        stable_source_key = source_id or f"{source_type}-anon-{ordinal:04d}"
        unit_id = "unit-maths-" + sha(
            f"{doc_id}|formal|{stable_source_key}|{m.start()}",
            24,
        )

        if source_id:
            source_id_to_unit[source_id] = unit_id

        sec = section_at(m.start(), sections)
        body = m.group("body").strip()
        title = extract_env_title(body)

        formal_units.append(
            {
                "unit_id": unit_id,
                "document_id": doc_id,
                "source_unit_id": source_id,
                "source_unit_type": source_type,
                "unit_type": source_type,
                "semantic_role": source_type,
                "title": title,
                "latex": body,
                "source_pos_start": m.start(),
                "source_pos_end": m.end(),
                "section_id": sec["section_id"] if sec else None,
                "section_path": sec["section_path"] if sec else [],
                "section_path_text": sec["section_path_text"] if sec else "",
                "depends_on_source_unit_id": depends_on,
                "labels": LABEL_RE.findall(body),
                "refs": REF_RE.findall(body),
                "quality_flags": [],
            }
        )

    # Build interstitial context units after \begin{document}, excluding the preamble.
    doc_begin = text.find(r"\begin{document}")
    if doc_begin >= 0:
        content_start = doc_begin + len(r"\begin{document}")
    else:
        content_start = 0
    doc_end = text.rfind(r"\end{document}")
    if doc_end < 0:
        doc_end = len(text)

    gaps = []
    cursor = content_start
    for m in formal_matches:
        if m.start() > cursor:
            gaps.append((cursor, m.start(), text[cursor:m.start()]))
        cursor = max(cursor, m.end())
    if cursor < doc_end:
        gaps.append((cursor, doc_end, text[cursor:doc_end]))

    implicit_units: list[dict] = []
    gap_counter = 0

    for start, end, raw_gap in gaps:
        body = normalize_gap(raw_gap)
        if not body:
            continue

        # Pure cue handled below as a semantic relation, not a separate retrieval chunk.
        if re.fullmatch(
            r"Le corollaire de ce théorème est alors\s*:",
            body.strip(),
            re.I,
        ):
            continue

        gap_counter += 1
        sec = section_at(start, sections)
        unit_type = classify_gap(body)
        unit_id = "unit-maths-" + sha(
            f"{doc_id}|implicit|{gap_counter}|{start}|{body[:128]}",
            24,
        )

        implicit_units.append(
            {
                "unit_id": unit_id,
                "document_id": doc_id,
                "source_unit_id": None,
                "source_unit_type": None,
                "unit_type": unit_type,
                "semantic_role": unit_type,
                "title": None,
                "latex": body,
                "source_pos_start": start,
                "source_pos_end": end,
                "section_id": sec["section_id"] if sec else None,
                "section_path": sec["section_path"] if sec else [],
                "section_path_text": sec["section_path_text"] if sec else "",
                "depends_on_source_unit_id": None,
                "labels": LABEL_RE.findall(body),
                "refs": REF_RE.findall(body),
                "quality_flags": [],
            }
        )

    units = sorted(
        formal_units + implicit_units,
        key=lambda u: u["source_pos_start"],
    )

    # Reclassify the explicitly announced corollary while preserving source_unit_type=theorem.
    for i, unit in enumerate(units):
        if unit.get("source_unit_id") != "theorem-0148":
            continue
        # Source itself says it is a corollary immediately before this theorem marker.
        prev_formal = None
        for j in range(i - 1, -1, -1):
            if units[j].get("source_unit_id"):
                prev_formal = units[j]
                break
        if prev_formal:
            between = text[
                prev_formal["source_pos_end"] : unit["source_pos_start"]
            ]
            if re.search(
                r"Le corollaire de ce théorème est alors",
                between,
                re.I,
            ):
                unit["semantic_role"] = "corollary"
                unit["corollary_of_unit_id"] = prev_formal["unit_id"]

    # Labels can belong to units, formulas, or sections.
    label_to_unit: dict[str, str] = {}
    for unit in units:
        for label in unit["labels"]:
            label_to_unit[label] = unit["unit_id"]

    label_to_section = {
        section["label"]: section["section_id"]
        for section in sections
        if section.get("label")
    }

    formulas: list[dict] = []
    for unit in units:
        fs = extract_formulas(unit["unit_id"], unit["latex"])
        formulas.extend(fs)
        unit["formula_ids"] = [f["formula_id"] for f in fs]

    label_to_formula = {}
    for formula in formulas:
        for label in formula["labels"]:
            label_to_formula[label] = formula["formula_id"]

    # Quality flags: preserve source and flag, never silently correct.
    quality_flags: list[dict] = []

    for unit in units:
        bal, min_bal = brace_balance(unit["latex"])
        if bal != 0 or min_bal < 0:
            qid = "qf-" + sha(
                f"{unit['unit_id']}|latex_brace_imbalance|{bal}|{min_bal}",
                24,
            )
            flag = {
                "quality_flag_id": qid,
                "unit_id": unit["unit_id"],
                "severity": "warning",
                "type": "latex_brace_imbalance",
                "details": {
                    "final_balance": bal,
                    "minimum_balance": min_bal,
                },
                "action": (
                    "Conserver la source telle quelle; ne pas corriger automatiquement. "
                    "Révision humaine recommandée avant usage comme autorité mathématique."
                ),
            }
            quality_flags.append(flag)
            unit["quality_flags"].append(qid)

    # One internal inconsistency is demonstrable from this source alone:
    # theorem-0001 states n(n-1)/2 but its attached proof concludes n(n+1)/2.
    theorem1 = next(
        (u for u in units if u.get("source_unit_id") == "theorem-0001"),
        None,
    )
    proof1 = next(
        (
            u
            for u in units
            if u.get("depends_on_source_unit_id") == "theorem-0001"
        ),
        None,
    )
    if theorem1 and proof1:
        if (
            "n(n-1)" in theorem1["latex"].replace(" ", "")
            and "n(n+1)" in proof1["latex"].replace(" ", "")
        ):
            for unit in (theorem1, proof1):
                qid = "qf-" + sha(
                    f"{unit['unit_id']}|internal_formula_conflict|theorem-0001",
                    24,
                )
                flag = {
                    "quality_flag_id": qid,
                    "unit_id": unit["unit_id"],
                    "severity": "critical_for_answering",
                    "type": "internal_formula_conflict",
                    "details": {
                        "source_statement": r"\sum_{k=1}^n k = \dfrac{n(n-1)}{2}",
                        "attached_proof_conclusion": r"S=\dfrac{n(n+1)}{2}",
                        "note": (
                            "Le document se contredit entre l'énoncé et sa propre preuve. "
                            "Aucune correction externe n'est appliquée automatiquement."
                        ),
                    },
                    "action": (
                        "Le retriever doit signaler le conflit et le générateur ne doit "
                        "pas présenter l'énoncé comme vérifié sans revue humaine."
                    ),
                }
                quality_flags.append(flag)
                unit["quality_flags"].append(qid)

    relations: list[dict] = []

    # Explicit proof dependencies.
    for unit in units:
        dep = unit.get("depends_on_source_unit_id")
        if dep:
            target = source_id_to_unit.get(dep)
            relations.append(
                make_relation(
                    "proof_of",
                    unit["unit_id"],
                    to_unit_id=target,
                    target_label=dep if target is None else None,
                    evidence=f"RAG_UNIT_BEGIN depends_on={dep}",
                    confidence="explicit",
                )
            )
            unit["proof_of"] = target
        else:
            unit["proof_of"] = None

    # Source-announced corollary.
    for unit in units:
        target = unit.get("corollary_of_unit_id")
        if target:
            relations.append(
                make_relation(
                    "corollary_of",
                    unit["unit_id"],
                    to_unit_id=target,
                    evidence="Texte source: « Le corollaire de ce théorème est alors : »",
                    confidence="explicit_text",
                )
            )

    # Explicit LaTeX cross references.
    for unit in units:
        for target_label in unit["refs"]:
            target_unit = label_to_unit.get(target_label)
            target_section = label_to_section.get(target_label)
            relations.append(
                make_relation(
                    "explicit_ref",
                    unit["unit_id"],
                    to_unit_id=target_unit,
                    target_label=target_label,
                    target_section_id=target_section,
                    evidence=f"Référence LaTeX explicite vers {target_label}",
                    confidence="explicit",
                )
            )

    # Formula label references are also explicit, and formula IDs stay on the chunk.
    formula_relations = []
    for formula in formulas:
        for target_label in formula["refs"]:
            target_unit = label_to_unit.get(target_label)
            if target_unit:
                formula_relations.append(
                    {
                        "from_formula_id": formula["formula_id"],
                        "to_unit_id": target_unit,
                        "target_label": target_label,
                        "type": "formula_explicit_ref",
                    }
                )

    # Structural adjacency inside the exact same section. This is soft metadata,
    # not a mathematical implication.
    by_section: dict[str, list[dict]] = {}
    for unit in units:
        sid = unit.get("section_id")
        if sid:
            by_section.setdefault(sid, []).append(unit)

    for section_units in by_section.values():
        section_units.sort(key=lambda u: u["source_pos_start"])
        for a, b in zip(section_units, section_units[1:]):
            relations.append(
                make_relation(
                    "next_in_section",
                    a["unit_id"],
                    to_unit_id=b["unit_id"],
                    evidence="Ordre documentaire dans la même section",
                    confidence="structural_only",
                )
            )

    # Remarks/warnings attach softly to immediately preceding formal unit when
    # no section boundary intervenes. Hypothesis notes are scoped at section level.
    unit_index = {u["unit_id"]: i for i, u in enumerate(units)}
    for i, unit in enumerate(units):
        if unit["unit_type"] in {"remark", "warning"}:
            prev = None
            for j in range(i - 1, -1, -1):
                cand = units[j]
                if cand.get("section_id") != unit.get("section_id"):
                    break
                if cand.get("source_unit_id"):
                    prev = cand
                    break
            if prev:
                relations.append(
                    make_relation(
                        "remark_about",
                        unit["unit_id"],
                        to_unit_id=prev["unit_id"],
                        evidence="Remarque/Attention immédiatement postérieure dans la même section",
                        confidence="structural_adjacent",
                    )
                )
                unit["remark_about"] = prev["unit_id"]

        if unit["unit_type"] == "hypothesis_note":
            unit["scope_section_id"] = unit.get("section_id")

    # Deduplicate exact relation edges while preserving how many times an explicit
    # edge occurred in the source. Repeated \\ref occurrences are evidence frequency,
    # not distinct graph edges.
    dedup_relations = {}
    for rel in relations:
        rid = rel["relation_id"]
        if rid in dedup_relations:
            dedup_relations[rid]["source_occurrences"] = (
                dedup_relations[rid].get("source_occurrences", 1) + 1
            )
        else:
            rel["source_occurrences"] = 1
            dedup_relations[rid] = rel
    relations = list(dedup_relations.values())

    # Unit local neighbor metadata and nearby definitions; do not call these dependencies.
    for section_units in by_section.values():
        section_units.sort(key=lambda u: u["source_pos_start"])
        for idx, unit in enumerate(section_units):
            unit["previous_unit_id"] = (
                section_units[idx - 1]["unit_id"] if idx > 0 else None
            )
            unit["next_unit_id"] = (
                section_units[idx + 1]["unit_id"]
                if idx + 1 < len(section_units)
                else None
            )
            preceding_defs = [
                u["unit_id"]
                for u in section_units[:idx]
                if u["unit_type"] == "definition"
            ]
            unit["nearby_preceding_definition_ids"] = preceding_defs[-3:]

    # Resolved relation targets / all refs sanity.
    unresolved_refs = []
    for unit in units:
        for target_label in unit["refs"]:
            if (
                target_label not in label_to_unit
                and target_label not in label_to_section
                and target_label not in label_to_formula
            ):
                unresolved_refs.append(
                    {
                        "unit_id": unit["unit_id"],
                        "target_label": target_label,
                    }
                )

    # Chunks: formal mathematical objects are atomic. In this corpus the largest
    # formal object is below the chosen max; prose gaps also fit under 6000 chars.
    chunks: list[dict] = []
    unit_by_id = {u["unit_id"]: u for u in units}
    quality_by_unit: dict[str, list[dict]] = {}
    for qf in quality_flags:
        quality_by_unit.setdefault(qf["unit_id"], []).append(qf)

    relation_targets: dict[str, set[str]] = {}
    for rel in relations:
        if rel.get("to_unit_id"):
            relation_targets.setdefault(
                rel["from_unit_id"], set()
            ).add(rel["to_unit_id"])

    for unit in units:
        context_lines = [
            "Corpus: Maths — recueil mathématique LaTeX fourni par l'utilisateur",
            f"Section: {unit.get('section_path_text') or '(racine)'}",
            f"Type: {unit['unit_type']}",
        ]

        if unit.get("semantic_role") != unit["unit_type"]:
            context_lines.append(
                f"Rôle sémantique explicite: {unit['semantic_role']}"
            )

        if unit.get("title"):
            context_lines.append(f"Titre: {unit['title']}")

        if unit.get("source_unit_id"):
            context_lines.append(
                f"Identifiant source: {unit['source_unit_id']}"
            )

        if unit["unit_type"] == "proof" and unit.get("proof_of"):
            parent = unit_by_id.get(unit["proof_of"])
            if parent:
                context_lines.append(
                    f"Preuve de: {parent.get('title') or parent.get('source_unit_id')}"
                )
                parent_context = plainish_latex(parent["latex"])[:1400]
                context_lines.append(
                    "Contexte de l'énoncé démontré:\n" + parent_context
                )

        if unit.get("semantic_role") == "corollary":
            parent = unit_by_id.get(unit.get("corollary_of_unit_id"))
            if parent:
                context_lines.append(
                    "Théorème parent annoncé par la source:\n"
                    + plainish_latex(parent["latex"])[:1200]
                )

        if unit.get("remark_about"):
            parent = unit_by_id.get(unit["remark_about"])
            if parent:
                context_lines.append(
                    "Objet mathématique commenté:\n"
                    + plainish_latex(parent["latex"])[:900]
                )

        unit_qflags = quality_by_unit.get(unit["unit_id"], [])
        if unit_qflags:
            warning_lines = []
            for qf in unit_qflags:
                warning_lines.append(
                    f"{qf['type']} ({qf['severity']}): "
                    + str(qf.get("details", {}).get("note", qf.get("details", "")))
                )
            context_lines.append(
                "Avertissement qualité de la source — ne pas corriger silencieusement:\n"
                + "\n".join(warning_lines)
            )

        context_lines.append(
            "Contenu:\n" + plainish_latex(unit["latex"])
        )

        embedding_text = "\n\n".join(context_lines)

        cid = "chunk-maths-" + sha(
            f"{unit['unit_id']}|{unit['latex']}",
            24,
        )
        qdrant_id = str(
            uuid.uuid5(
                DOC_NAMESPACE,
                cid,
            )
        )

        chunk = {
            "chunk_id": cid,
            "qdrant_id": qdrant_id,
            "document_id": doc_id,
            "unit_id": unit["unit_id"],
            "source_unit_id": unit.get("source_unit_id"),
            "content_kind": "math_reference",
            "unit_type": unit["unit_type"],
            "semantic_role": unit.get("semantic_role"),
            "title": unit.get("title"),
            "section_id": unit.get("section_id"),
            "section_path": unit.get("section_path", []),
            "section_path_text": unit.get("section_path_text", ""),
            "proof_of": unit.get("proof_of"),
            "corollary_of": unit.get("corollary_of_unit_id"),
            "remark_about": unit.get("remark_about"),
            "scope_section_id": unit.get("scope_section_id"),
            "related_unit_ids": sorted(
                relation_targets.get(unit["unit_id"], set())
            ),
            "nearby_preceding_definition_ids": unit.get(
                "nearby_preceding_definition_ids", []
            ),
            "previous_unit_id": unit.get("previous_unit_id"),
            "next_unit_id": unit.get("next_unit_id"),
            "formula_ids": unit.get("formula_ids", []),
            "labels": unit.get("labels", []),
            "refs": unit.get("refs", []),
            "quality_flags": unit.get("quality_flags", []),
            "quality_flag_details": [
                {
                    "quality_flag_id": qf["quality_flag_id"],
                    "severity": qf["severity"],
                    "type": qf["type"],
                    "details": qf.get("details", {}),
                    "action": qf.get("action"),
                }
                for qf in unit_qflags
            ],
            "requires_human_math_check": any(
                qf["quality_flag_id"] in unit.get("quality_flags", [])
                and qf["severity"] == "critical_for_answering"
                for qf in quality_flags
            ),
            "embedding_text": embedding_text,
            "latex": unit["latex"],
            "content_hash": hashlib.sha256(
                unit["latex"].encode("utf-8")
            ).hexdigest(),
            "source_file": source.name,
            "language": "fr",
        }
        chunks.append(chunk)

    # Macro definitions in preamble.
    macros = []
    for m in re.finditer(
        r"\\newcommand\s*\{\\([A-Za-z@]+)\}(?:\[(\d+)\])?\s*\{",
        text,
    ):
        body, end = parse_balanced(text, m.end() - 1)
        macros.append(
            {
                "document_id": doc_id,
                "command": m.group(1),
                "num_args": int(m.group(2) or 0),
                "definition": body,
            }
        )

    document = {
        "document_id": doc_id,
        "title": "Maths — recueil mathématique LaTeX",
        "language": "fr",
        "source_file": source.name,
        "source_sha256": source_sha,
        "source_chars": len(text),
        "source_bytes": len(raw_bytes),
        "content_kind": "math_reference",
        "rag_sanitized_source": "RAG_SANITIZED_SOURCE" in text,
        "complete_inputs": True,
        "macro_context_complete": True,
    }

    # Source-level warning derived only from internal consistency / syntax.
    document_quality = {
        "document_id": doc_id,
        "unresolved_refs": len(unresolved_refs),
        "formal_proofs": sum(
            1 for u in units if u["unit_type"] == "proof"
        ),
        "linked_proofs": sum(
            1
            for u in units
            if u["unit_type"] == "proof" and u.get("proof_of")
        ),
        "units_with_quality_flags": sum(
            bool(u.get("quality_flags"))
            for u in units
        ),
    }

    stats = {
        "documents": 1,
        "sections": len(sections),
        "units": len(units),
        "formal_units": len(formal_units),
        "implicit_context_units": len(implicit_units),
        "chunks": len(chunks),
        "unit_types": dict(
            Counter(u["unit_type"] for u in units)
        ),
        "semantic_roles": dict(
            Counter(u["semantic_role"] for u in units)
        ),
        "formulas": len(formulas),
        "labels": len(LABEL_RE.findall(text)),
        "unique_labels": len(set(LABEL_RE.findall(text))),
        "refs": len(REF_RE.findall(text)),
        "unresolved_refs": len(unresolved_refs),
        "relations": len(relations),
        "relation_types": dict(
            Counter(r["type"] for r in relations)
        ),
        "proofs": document_quality["formal_proofs"],
        "linked_proofs": document_quality["linked_proofs"],
        "quality_flags": len(quality_flags),
        "quality_flag_types": dict(
            Counter(q["type"] for q in quality_flags)
        ),
        "max_chunk_latex_chars": max(
            (len(c["latex"]) for c in chunks),
            default=0,
        ),
        "chunks_over_6000": sum(
            len(c["latex"]) > 6000
            for c in chunks
        ),
        "source_sha256": source_sha,
    }

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "documents.jsonl", [document])
    write_jsonl(out / "sections.jsonl", sections)
    write_jsonl(out / "macros.jsonl", macros)
    write_jsonl(out / "units.jsonl", units)
    write_jsonl(out / "formulas.jsonl", formulas)
    write_jsonl(out / "formula_relations.jsonl", formula_relations)
    write_jsonl(out / "relations.jsonl", relations)
    write_jsonl(out / "quality_flags.jsonl", quality_flags)
    write_jsonl(out / "unresolved_refs.jsonl", unresolved_refs)
    write_jsonl(out / "chunks.jsonl", chunks)
    (out / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    stats = build(args.source, args.out)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
