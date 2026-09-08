from pydantic import BaseModel, Field, AliasChoices
from typing import Annotated
import numpy as np
from enum import Enum

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class Approach(str, Enum):
    KWISE = "kwise"
    ROWWISE = "rowwise"
    FULL = "full"

class Chunk(BaseModel):
    text: str
    metadata: dict
    embedding: list[float]

class SimpleVectorStore(BaseModel):
    name: str
    chunks: list[Chunk]

    def add_chunk(self, chunk: Chunk):
        self.chunks.append(chunk)

    def get_top_chunks(self, query: list[float], top_k: int = 5):
        sorted_chunks = sorted(
            self.chunks,
            key=lambda c: cosine(c.embedding, query),
            reverse=True,
        )
        return sorted_chunks[:min(top_k, len(sorted_chunks))]

class Factor(BaseModel):
    name: str
    description: Annotated[str | None, Field(default=None)]

class Factors(BaseModel):
    factors: list[Factor]

class EmbeddedFactor(Factor):
    embedding: list[float]

class EmbeddedFactors(BaseModel):
    factors: list[EmbeddedFactor]

class SSIMFullGraphElement(BaseModel):
    factor_i: str
    factor_j: str
    relationship: str

class SSIMFullGraphElements(BaseModel):
    factors: list[SSIMFullGraphElement]

class SSIMKWiseElement(BaseModel):
    provided_factor_name: str
    factor_name: str
    relationship: str

class SSIMKWiseElements(BaseModel):
    factors: list[SSIMKWiseElement]

class StructuralSelfInteractionElement(BaseModel):
    factor_i: Annotated[
        str,
        Field(
            validation_alias=AliasChoices("factor_i", "provided_factor_name")
        ),
    ]
    factor_j: Annotated[
        str,
        Field(validation_alias=AliasChoices("factor_j", "factor_name")),
    ]
    relationship: str

class StructuralSelfInteractionElements(BaseModel):
    factors: list[StructuralSelfInteractionElement]
