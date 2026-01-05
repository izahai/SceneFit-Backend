# app/schemas/basis_sch.py

from pydantic import BaseModel
from typing import List


class RetrievalItem(BaseModel):
    name_clothes: str
    positive_score: float
    negative_score: float
    confidence: float


class RetrievalResponse(BaseModel):
    count: int
    results: List[RetrievalItem]
