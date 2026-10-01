"""Catalogue commun aux scripts historiques et au futur RAG.

Le schéma canonique reprend les champs déjà présents dans sources.yaml :
id, title, author, year, lang, domain, level, type, topics, file, source_url.
Les chemins sont relatifs à PROJECT_ROOT et doivent rester dans corpus/raw.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import CATALOG_PATH, PROJECT_ROOT, RAW_DIR


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    title: str = Field(min_length=3)
    author: str | None = None
    year: int | None = Field(default=None, ge=1800, le=2100)
    lang: Literal["fr", "en"]
    domain: Literal["mathematiques", "cryptographie"]
    level: Literal["L1", "L2", "PREPA", "L2_PREPA", "L3", "RECHERCHE"]
    type: Literal["cours", "article", "exercice", "corrige", "concours", "standard", "these"]
    topics: list[str] = Field(min_length=1)
    file: str
    source_url: str | None = None
    license: str | None = None
    rights_status: Literal["autorise", "non_verifie", "interdit"] = "non_verifie"
    authority_level: Literal["officiel", "universitaire", "recherche", "complementaire"] | None = None
    trusted: bool = False
    # L'admission est distincte de la simple présence au catalogue.
    admission: Literal["catalogue", "a_controler", "admis", "rejete"] = "catalogue"
    book_selected: bool = False

    @field_validator("file")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        path = Path(value)
        if (path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".pdf"
                or path.parts[:2] != ("corpus", "raw")):
            raise ValueError("file doit être un PDF relatif sous corpus/raw/")
        return value

    @model_validator(mode="after")
    def validate_trust(self) -> "Source":
        if self.trusted and self.rights_status != "autorise":
            raise ValueError("trusted exige rights_status=autorise")
        if self.admission == "admis" and self.rights_status == "interdit":
            raise ValueError("une source interdite ne peut pas être admise")
        return self

    @property
    def language(self) -> str:
        return self.lang

    @property
    def document_type(self) -> str:
        return self.type

    def pdf_path(self, root: Path = PROJECT_ROOT) -> Path:
        root = root.resolve()
        raw = (root / "corpus" / "raw").resolve()
        pdf = (root / self.file).resolve()
        if not pdf.is_relative_to(raw) or pdf == raw:
            raise ValueError(f"chemin PDF hors corpus/raw : {self.file}")
        return pdf


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: list[Source]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "Catalog":
        ids = [source.id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("identifiants de sources dupliqués")
        return self


def load_catalog(path: Path = CATALOG_PATH, require_files: bool = False,
                 root: Path = PROJECT_ROOT) -> Catalog:
    if not path.exists():
        raise FileNotFoundError(f"Catalogue introuvable : {path}")
    catalog = Catalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    missing = []
    for source in catalog.sources:
        pdf = source.pdf_path(root)
        if require_files and not pdf.is_file():
            missing.append(str(pdf))
    if missing:
        raise FileNotFoundError("Sources absentes :\n" + "\n".join(missing))
    return catalog
