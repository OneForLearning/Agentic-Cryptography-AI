# Livre_Crypto — Pipeline RAG structurel pour corpus LaTeX mathématique

Ce kit est conçu pour le corpus LaTeX fourni. Il ne traite pas un cours mathématique comme une simple suite de tokens : il conserve la hiérarchie du document, les environnements formels, les preuves, les labels/références et les macros locales.

## 1. Ce que le pipeline conserve

Pour chaque document : titre, auteur, classe LaTeX, fichiers inclus, `\\input` manquants, complétude du contexte de macros.

Pour chaque unité : `definition`, `lemma`, `proposition`, `theorem`, `corollary`, `proof`, `claim`, `notation`, `remark`, `example`, `problem`, `exercise`, `method`, `pitfall`, prose, section courante, labels, références, macros utilisées, lignes source et LaTeX brut.

Les preuves sont vectorisées séparément de l'énoncé afin de préserver le signal sémantique, mais reliées par `proof_of`. Le retrieval réassemble énoncé et preuve avant de construire le contexte du LLM.

## 2. Installation dans `~/Livre_Crypto`

Copier le contenu de ce kit dans le projet, en conservant `scripts/`, `config/` et `schemas/`.

```bash
cd ~/Livre_Crypto
uv add pyyaml ollama qdrant-client
```

Qdrant local :

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

Vérification :

```bash
curl http://127.0.0.1:6333/healthz
```

Le modèle d'embedding prévu est celui déjà utilisé dans le projet :

```bash
ollama list | grep bge-m3
```

## 3. Préparer l'archive

Placer l'archive dans le projet, par exemple :

```text
corpus/incoming/latex_deja_traite.zip
```

Puis :

```bash
uv run python scripts/prepare_latex_corpus.py \
  corpus/incoming/latex_deja_traite.zip \
  --out corpus/raw/latex/expanded \
  --clean
```

Le script extrait récursivement les ZIP internes et ignore `__MACOSX` / `.DS_Store`.

## 4. Extraction structurée

```bash
uv run python scripts/extract_latex_structured.py \
  --latex-root corpus/raw/latex/expanded \
  --out corpus/processed/latex
```

Sorties :

```text
corpus/processed/latex/
├── documents.jsonl
├── macros.jsonl
├── units.jsonl
├── relations.jsonl
└── extraction_stats.json
```

`relations.jsonl` contient notamment `proof_of`, `references` et `next`.

## 5. Chunking mathématique

```bash
uv run python scripts/build_latex_chunks.py \
  --processed corpus/processed/latex \
  --target-chars 4500 \
  --max-chars 6000 \
  --prose-overlap 500
```

Politique :

- un énoncé formel reste atomique tant que possible ;
- aucune fenêtre glissante n'est appliquée entre un théorème et sa preuve ;
- les preuves longues sont divisées, mais chaque morceau garde le même `proof_of` ;
- l'overlap est réservé à la prose ;
- les environnements LaTeX imbriqués sont évités comme points de coupure ;
- `embedding_text` reçoit le breadcrumb de section et les définitions des macros utilisées ;
- `latex` reste la source de vérité.

## 6. Validation avant embedding

```bash
uv run python scripts/validate_latex_rag.py \
  --processed corpus/processed/latex
```

Ne pas utiliser `--strict` tant que les cours incomplets identifiés dans l'analyse n'ont pas été complétés.

## 7. Embeddings + Qdrant + index lexical

Vérifier que Qdrant et Ollama tournent, puis :

```bash
uv run python scripts/index_latex_rag.py \
  --processed corpus/processed/latex \
  --model bge-m3 \
  --collection livre_crypto_latex_v1 \
  --recreate
```

Le script :

1. appelle `bge-m3` via Ollama ;
2. détecte automatiquement la dimension du vecteur ;
3. crée la collection Qdrant ;
4. indexe les payloads structurels ;
5. construit `corpus/processed/latex/lexical.db` avec SQLite FTS5.

L'index lexical est important pour les noms de théorèmes, notations exactes, labels et termes mathématiques que la similarité dense peut sous-pondérer.

## 8. Retrieval hybride

```bash
uv run python scripts/retrieve_latex.py \
  "Démontre le théorème des intervalles emboîtés"
```

Le retrieval effectue :

```text
requête
  ├─ bge-m3 → Qdrant dense top-k
  └─ SQLite FTS5 → lexical top-k
             ↓
        RRF fusion
             ↓
   bonus selon type demandé
             ↓
 déduplication content_hash
             ↓
 expansion du graphe
 theorem ↔ proof, références
             ↓
 contexte final
```

Pour récupérer le JSON :

```bash
uv run python scripts/retrieve_latex.py \
  "définition d'un sous-groupe normal" \
  --json
```

## 9. Envoyer le contexte au LLM

Avec le modèle actuel :

```bash
uv run python scripts/ask_latex_rag.py \
  "Explique le théorème de Lagrange et donne sa preuve" \
  --llm qwen3:14b
```

Plus tard, remplacer simplement `qwen3:14b` par le nom final de `cryptoExpert`.

## 10. Schéma logique d'un chunk

```json
{
  "chunk_id": "chunk-...",
  "document_id": "doc-...",
  "course_title": "Analysis I...",
  "unit_type": "theorem",
  "title": "Nested Interval Theorem",
  "label": "...",
  "section_path": [
    {"kind": "section", "title": "Numerical Sequences and Series"},
    {"kind": "subsection", "title": "Bolzano--Weierstrass Theorem"}
  ],
  "proof_of": null,
  "related_unit_ids": ["unit-..."],
  "references": [],
  "macros_used": ["NN", "RR", "abs"],
  "macro_context": [
    {"command": "RR", "definition": "\\mathbb R", "num_args": 0}
  ],
  "embedding_text": "Course: ...",
  "latex": "\\begin{theorem}...",
  "content_hash": "..."
}
```

## 11. Règles à conserver lors de l'évolution du RAG

1. Ne jamais remplacer le LaTeX brut par sa seule conversion en texte.
2. Ne jamais couper mécaniquement au milieu d'un environnement `theorem`, `proof`, `align`, `enumerate`, etc.
3. Ne jamais perdre `proof_of` lors d'un rechunking.
4. Les labels doivent être scoppés au document ; le corpus possède des labels identiques dans des cours distincts.
5. Une macro non définie dans le document ne doit pas être devinée.
6. Conserver les documents incomplets, mais avec `complete_inputs=false`.
7. Dédupliquer le contexte final par `content_hash`, pas les sources dans le stockage : la provenance de chaque édition doit rester disponible.
8. Faire la séparation entre le texte utilisé pour l'embedding et le LaTeX utilisé comme preuve documentaire.
9. Réindexer dans une nouvelle collection (`livre_crypto_latex_v2`, etc.) lorsqu'un changement de schéma ou de chunking est important ; ne pas muter silencieusement un index de référence.
10. Valider le retrieval avec un benchmark de questions avant de brancher le RAG sur le fine-tuning ou sur la génération du livre.

## 12. Limites détectées dans l'archive actuelle

Le rapport complet se trouve dans `analysis/CORPUS_ANALYSIS.md`. Les principales limites sont :

- certains cours fondés sur `\\input` sont incomplets ;
- `abstract-algebra-notes.tex` référence un `preamble.tex` absent ;
- certains documents sont des variantes ou éditions très proches ;
- les macros sont nombreuses et doivent rester document-scopées.

Ces limites ne bloquent pas l'indexation, mais doivent rester visibles dans les métadonnées et dans l'évaluation des réponses du LLM.
