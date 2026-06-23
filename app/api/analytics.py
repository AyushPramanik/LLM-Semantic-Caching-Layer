"""Analytics endpoints for threshold tuning."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.analytics.threshold import (
    DEFAULT_THRESHOLDS,
    ThresholdAnalyzer,
    ThresholdReport,
    ThresholdTestCase,
)
from app.analytics.near_miss import NearMissReport, NearMissTracker
from app.api.dependencies import get_embeddings
from app.embeddings.base import EmbeddingService
from app.policies.threshold_engine import AdaptiveThresholdEngine, ThresholdCategory

router = APIRouter(prefix="/analytics", tags=["analytics"])


def get_threshold_engine(request: Request) -> AdaptiveThresholdEngine:
    return request.app.state.threshold_engine


def get_near_miss_tracker(request: Request) -> NearMissTracker:
    return request.app.state.near_miss_tracker


@router.get("/near-misses", response_model=NearMissReport)
async def near_misses(
    tracker: NearMissTracker = Depends(get_near_miss_tracker),
) -> NearMissReport:
    return tracker.report()


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


class ThresholdFeedback(BaseModel):
    category: ThresholdCategory
    false_hit: bool = Field(
        ..., description="True if the cache served a wrong answer (tighten); "
        "False if an equivalent prompt was needlessly missed (loosen)."
    )


@router.get("/thresholds")
async def current_thresholds(
    engine: AdaptiveThresholdEngine = Depends(get_threshold_engine),
) -> dict[str, float]:
    return engine.snapshot()


@router.post("/threshold-feedback")
async def threshold_feedback(
    feedback: ThresholdFeedback,
    engine: AdaptiveThresholdEngine = Depends(get_threshold_engine),
) -> dict[str, object]:
    updated = engine.record_feedback(feedback.category, false_hit=feedback.false_hit)
    return {"category": feedback.category.value, "threshold": updated}
