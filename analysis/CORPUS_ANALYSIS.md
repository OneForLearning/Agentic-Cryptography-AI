# Analyse du corpus LaTeX pour le RAG Livre_Crypto
## Résumé quantitatif
- 12 archives LaTeX internes analysées.
- 87 fichiers `.tex` utiles.
- 36 documents racines (`\documentclass`).
- 12,8 Mo environ de LaTeX.
- 15 258 unités structurées extraites.
- 15 552 chunks RAG produits avec la politique proposée.
- 777 définitions de macros, correspondant à 286 noms de macros personnalisées distincts.
- 3219 références LaTeX (`\ref`, `\eqref`, etc.), dont 3186 résolues dans le document (99.0%).
- 2650 unités `proof`, dont 2633 rattachées automatiquement à un énoncé (99.4%).
## Densité mathématique
- `definition` : 1525
- `theorem` : 1386
- `proposition` : 807
- `lemma` : 367
- `corollary` : 321
- `proof` : 2650
- `claim` : 29
- `notation` : 34
- `remark` : 703
- `example` : 3526
- `example_set` : 74
- `method` : 270
- `pitfall` : 362
- `exercise_set` : 856

Cette densité montre qu’un découpage fixe par nombre de caractères ou de tokens détruirait une information structurelle importante. Les preuves suivent très souvent leur énoncé : le lien énoncé→preuve doit être conservé comme relation explicite, pas seulement par voisinage vectoriel.

## Points de qualité détectés
- 4 documents ont des `\input`/`\include` manquants. Ils restent indexables, mais doivent porter `complete_inputs=false`.
  - **Introduction to Abstract~Algebra** : Lectures/1, Lectures/10, Lectures/21, Lectures/22, Lectures/24, Lectures/33
  - **abstract algebra notes** : preamble.tex
  - **Fields and Galois Theory** : Lectures/8, Lectures/12, Lectures/13, Lectures/16, Lectures/17
  - **\lsstyle Number Theory** : Lectures/1, Lectures/3, Lectures/8, Lectures/9, Lectures/12
- 1 document a un contexte de macros incomplet : **abstract algebra notes** (préambule absent). Aucune expansion conjecturale ne doit être appliquée.
- 41 groupes de chunks exactement identiques apparaissent dans plusieurs documents (87 occurrences). Le retrieval doit dédupliquer sur `content_hash`.
- Recouvrement documentaire notable : les deux cours de probabilité sont très proches; les deux versions de Logic and Set Theory partagent également du contenu. Il faut conserver les sources séparées pour la provenance, mais limiter les doublons dans le contexte final.

## Architecture recommandée
1. **LaTeX brut = source de vérité.** Ne pas convertir le corpus en texte plat puis jeter le LaTeX. Chaque chunk conserve `latex` et une seconde vue `embedding_text`.
2. **Extraction structurelle.** Hiérarchie `part/chapter/section/subsection`, unités formelles, labels, références, macros et relations énoncé-preuve.
3. **Chunks atomiques par objet mathématique.** Une définition, un lemme, une proposition, un théorème ou un corollaire reste entier tant que sa taille le permet. Une preuve longue peut être divisée en parties, toutes liées par `proof_of`.
4. **Macros.** Conserver un registre par document. Pour l’embedding, ajouter seulement les définitions des macros effectivement utilisées dans le chunk; ne pas faire d’expansion agressive.
5. **Dense + lexical.** `bge-m3` via Ollama pour la similarité sémantique, SQLite FTS5 pour les noms exacts, symboles textuels, labels et terminologie. Fusion par Reciprocal Rank Fusion.
6. **Qdrant.** Un point par chunk, vecteur dense et payload riche (document, type, section, label, `proof_of`, hash).
7. **Expansion structurelle après retrieval.** Si un théorème est trouvé, joindre sa preuve; si une preuve est trouvée, joindre l’énoncé; suivre les références utiles dans la limite du budget de contexte.
8. **Déduplication.** Dédupliquer par `content_hash` et plafonner le nombre de chunks par document afin qu’une édition quasi-dupliquée ne monopolise pas le contexte.

## Politique de chunking proposée
- Cible prose : ~4 500 caractères; maximum : 6 000; overlap uniquement sur prose : 500 caractères.
- Définition/théorème/lemme/proposition/corollaire : unité atomique; pas d’overlap artificiel.
- Preuve : unité séparée avec `proof_of`; découpage uniquement si elle dépasse la limite, en respectant paragraphes et environnements LaTeX.
- Exemple/exercice/méthode/pitfall : unités indépendantes, utiles pour les questions pédagogiques.
- Le breadcrumb du cours et de la section est ajouté à `embedding_text`, jamais au LaTeX brut.

## Pourquoi ne pas concaténer théorème + preuve dans un seul vecteur ?
Parce qu’une preuve longue peut diluer le signal sémantique de l’énoncé. Le meilleur compromis est de vectoriser séparément l’énoncé et la preuve, puis de les relier. Le retrieval peut ainsi trouver directement le théorème pour une question de définition ou directement la preuve pour une question de démonstration, tout en reconstituant le couple avant envoi au LLM.

## Sorties de validation générées sur cette archive
- `documents.jsonl` : 36 documents avec complétude et provenance.
- `macros.jsonl` : registre des macros document-scopé.
- `units.jsonl` : 15 258 unités structurelles.
- `relations.jsonl` : graphe `proof_of`, `references`, `next`.
- `chunks.jsonl` : 15 552 chunks prêts pour embedding.
- Validation : **0 erreur structurelle**, 2 avertissements attendus (inputs manquants, préambule manquant).
