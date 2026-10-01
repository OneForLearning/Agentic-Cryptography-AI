from pathlib import Path

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)

CORPUS_DIR = PROJECT_ROOT / "corpus"
RAW_DIR = CORPUS_DIR / "raw"
PROCESSED_DIR = CORPUS_DIR / "processed"
CATALOG_PATH = CORPUS_DIR / "catalog" / "sources.yaml"

RAG_DIR = PROJECT_ROOT / "rag"
CHUNKS_DIR = RAG_DIR / "chunks"
INDEX_DIR = RAG_DIR / "index" / "qdrant"
REPORTS_DIR = RAG_DIR / "reports"

EMBEDDING_MODEL = "bge-m3"
MATH_MODEL = "mathematicien-14b"
CRYPTO_MODEL = "cryptographe-14b"
COLLECTION_NAME = "livre_crypto"
