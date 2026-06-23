"""Tests for the near-miss analyzer."""

from __future__ import annotations

from app.analytics.near_miss import NearMissTracker


def test_empty_report():
    tracker = NearMissTracker()
    report = tracker.report()
    assert report.count == 0
    assert "No near misses" in report.recommendation


def test_records_and_buckets_near_misses():
    tracker = NearMissTracker(min_samples=5)
    for score in [0.905, 0.915, 0.925, 0.935, 0.945, 0.948]:
        tracker.record(score=score, threshold=0.95, namespace="ns", prompt="x")
    report = tracker.report()
    assert report.count == 6
    assert sum(report.histogram.values()) == 6
    assert report.histogram["0.94-0.95"] >= 2


def test_recommendation_appears_after_min_samples():
    tracker = NearMissTracker(min_samples=20)
    for _ in range(25):
        tracker.record(score=0.93, threshold=0.95, namespace="ns", prompt="prompt")
    report = tracker.report()
    assert report.recommended_threshold is not None
    assert report.recommended_threshold < 0.95


def test_capacity_is_bounded():
    tracker = NearMissTracker(capacity=10)
    for i in range(50):
        tracker.record(score=0.92, threshold=0.95, namespace="ns", prompt=str(i))
    assert tracker.report().count == 10


def test_near_miss_endpoint(client):
    resp = client.get("/analytics/near-misses")
    assert resp.status_code == 200
    assert "count" in resp.json()
