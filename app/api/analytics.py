"""Analytics endpoints for threshold tuning."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.analytics.threshold import (
    DEFAULT_THRESHOLDS,
    ThresholdAnalyzer,
    ThresholdReport,
    ThresholdTestCase,
)
from app.api.dependencies import get_embeddings
from app.embeddings.base import EmbeddingService

router = APIRouter(prefix="/analytics", tags=["analytics"])


class ThresholdTestRequest(BaseModel):
    cases: list[ThresholdTestCase] = Field(..., min_length=1)
    thresholds: list[float] | None = None


@router.post("/threshold-test", response_model=ThresholdReport)
async def threshold_test(
    payload: ThresholdTestRequest,
    embeddings: EmbeddingService = Depends(get_embeddings),
) -> ThresholdReport:
    thresholds = tuple(payload.thresholds) if payload.thresholds else DEFAULT_THRESHOLDS
    analyzer = ThresholdAnalyzer(embedding_service=embeddings, thresholds=thresholds)
    return await analyzer.evaluate(payload.cases)
