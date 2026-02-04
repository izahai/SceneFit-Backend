from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.method_scorer import score_methods
from app.services.study_storage import append_response, read_all_payloads


router = APIRouter()


DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data")
DEFAULT_JSONL_PATH = os.path.join(DEFAULT_DATA_DIR, "user_study_responses.jsonl")


class UnityMethodResponse(BaseModel):
    methodId: str
    selectedRank: int = Field(..., description="0-based index into the method's 5 outfits (0..4 by default)")
    viewCounts: Optional[List[int]] = Field(
        None,
        description="Optional per-outfit counts of 'View' button clicks (length == num_outfits).",
    )


class UnityParticipantPayload(BaseModel):
    participantId: str
    timestamp: Optional[str] = None
    responses: List[UnityMethodResponse]
    finalWinnerMethodId: str


class StudyScoreQuery(BaseModel):
    # Hard-coded method ids/names for this study. Kept as a field so the
    # frontend can omit it or the backend can validate it.
    methods: Optional[List[str]] = Field(
        None,
        description="Optional override. If omitted, backend uses the study's fixed method names.",
    )
    alpha: float = Field(0.6, ge=0.0, le=1.0)
    num_outfits: int = Field(5, ge=1)


STUDY_METHODS = [
    "Image Editing",
    "Vision Language Model",
    "CLIP Model",
    "Asthetic Model",
]


@router.post("/study/response")
def submit_study_response(payload: UnityParticipantPayload) -> Dict[str, Any]:
    """Ingest a single participant response from Unity.

    This endpoint stores the raw participant payload as JSONL (append-only).
    """
    # Basic validation: ensure methodIds are unique per participant
    method_ids = [r.methodId for r in payload.responses]
    if len(set(method_ids)) != len(method_ids):
        raise HTTPException(status_code=400, detail="Duplicate methodId in responses")

    # Store the payload exactly (as dict) so we can re-score later.
    meta = append_response(payload.model_dump(), file_path=DEFAULT_JSONL_PATH)

    return {
        "ok": True,
        "stored": meta,
    }


@router.post("/study/score")
def get_study_score(query: StudyScoreQuery) -> Dict[str, Any]:
    """Compute aggregated scores from all stored participant payloads."""
    payloads = read_all_payloads(file_path=DEFAULT_JSONL_PATH)

    if not payloads:
        return {
            "methods": {},
            "summary": {
                "total_participants": 0,
                "ranked_methods": [],
                "alpha": query.alpha,
                "num_outfits": query.num_outfits,
                "storage_path": DEFAULT_JSONL_PATH,
            },
        }

    methods = query.methods or STUDY_METHODS
    result = score_methods(methods, payloads, alpha=query.alpha, num_outfits=query.num_outfits)

    result.setdefault("summary", {})
    result["summary"].update({
        "alpha": query.alpha,
        "num_outfits": query.num_outfits,
        "storage_path": DEFAULT_JSONL_PATH,
    })

    return result
