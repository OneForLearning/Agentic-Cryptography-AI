# Livre_Crypto — 6A.3 : catalogue et conservation de 6A.2

Ce paquet contient un remplacement pour `src/livre_crypto_rag/catalog.py`, un
nouveau script `scripts/snapshot_6a2.py` et un test ciblé. Aucun des PDF,
chunks, décisions ou exemples approuvés n'est modifié.

## Application sur le Mac

1. Décompresser le paquet hors de `~/Livre_Crypto`.
2. Copier uniquement `scripts/snapshot_6a2.py` dans `~/Livre_Crypto/scripts/`.
3. Depuis `~/Livre_Crypto`, exécuter :

   ```bash
   uv run python scripts/snapshot_6a2.py create
   uv run python scripts/snapshot_6a2.py verify snapshots/6A2-*.zip --compare-live
   ```

   Si plusieurs snapshots existent, donner le nom exact à `verify`. Le script
   refuse les entrées manquantes et n'écrase jamais une sauvegarde existante.
   Garder une copie indépendante du ZIP produit avant de remplacer du code.

4. Copier ensuite `src/livre_crypto_rag/catalog.py` dans le projet. Ne pas
   écraser `corpus/catalog/sources.yaml`. Vérifier :

   ```bash
   uv run python -c 'from livre_crypto_rag.catalog import load_catalog; c=load_catalog(require_files=True); print([(s.id,str(s.pdf_path())) for s in c.sources])'
   ```

5. Exécuter le test ciblé (si les dépendances du projet sont installées) :

   ```bash
   uv run python -m unittest discover -s tests -p test_catalog_snapshot.py
   ```

## Contrat du catalogue

Le YAML actuel est la convention canonique : `lang`, `type`, et `file` relatif
à la racine du projet sous `corpus/raw/`. Les identifiants existants, même en
minuscules, sont conservés. Les nouveaux champs facultatifs `rights_status`,
`admission` et `book_selected` ne rendent pas les deux PDF actuellement admis
par défaut : une source non documentée reste `catalogue` et `non_verifie`.

Le chargement valide la syntaxe des chemins et rejette une résolution hors de
`corpus/raw`, y compris par lien symbolique. `require_files=True` contrôle la
présence des PDF. Ce patch ne change pas `scripts/extract_corpus.py`, qui lit
encore directement le YAML ; la prochaine étape devra le faire passer par le
catalogue validé sans toucher aux sorties 6A.2.

## Limite

Le snapshot protège la cohérence des fichiers listés dans le manifeste et des
PDF déclarés. Il ne prouve pas la correction mathématique du contenu ni la
qualité des formules extraites. Le conserver hors du projet pour résister à une
perte du dossier de travail.
# AI-CRYPTOGRAPHIC-EXPERT
