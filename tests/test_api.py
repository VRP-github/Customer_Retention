from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

MOCK_PROBA = np.array([[0.3, 0.7]])


def make_mock_model():
    model = MagicMock()
    model.predict.side_effect = lambda df: np.ones(len(df), dtype=int)
    impl = MagicMock()
    impl.predict_proba.side_effect = lambda df: np.tile([0.3, 0.7], (len(df), 1))
    model._model_impl = impl
    return model


@pytest.fixture(scope="module")
def client():
    with patch("mlflow.pyfunc.load_model", return_value=make_mock_model()):
        from api import app, model_store
        model_store["model"] = make_mock_model()
        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("healthy", "degraded")
        assert "model_loaded" in body


class TestPredictSingle:
    VALID_PAYLOAD = {
        "Frequency": 10,
        "Monetary": 2500.0,
        "F_Score": 4,
        "M_Score": 5,
    }

    def test_valid_request_returns_200(self, client):
        r = client.post("/predict", json=self.VALID_PAYLOAD)
        assert r.status_code == 200

    def test_response_schema(self, client):
        r = client.post("/predict", json=self.VALID_PAYLOAD)
        body = r.json()
        pred = body["prediction"]
        assert "churn_probability" in pred
        assert "churn_label" in pred
        assert "risk_tier" in pred
        assert 0.0 <= pred["churn_probability"] <= 1.0

    def test_risk_tier_high_probability(self, client):
        r = client.post("/predict", json=self.VALID_PAYLOAD)
        tier = r.json()["prediction"]["risk_tier"]
        assert tier in ("High", "Critical")

    def test_invalid_frequency_zero(self, client):
        payload = {**self.VALID_PAYLOAD, "Frequency": 0}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_invalid_monetary_negative(self, client):
        payload = {**self.VALID_PAYLOAD, "Monetary": -100.0}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_f_score_out_of_range(self, client):
        payload = {**self.VALID_PAYLOAD, "F_Score": 6}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_missing_field(self, client):
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "M_Score"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422


class TestPredictBatch:
    def _make_batch(self, n: int) -> dict:
        return {
            "customers": [
                {"Frequency": 5, "Monetary": 500.0, "F_Score": 3, "M_Score": 3}
                for _ in range(n)
            ]
        }

    def test_batch_returns_correct_count(self, client):
        payload = self._make_batch(5)
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 5
        assert len(body["predictions"]) == 5

    def test_empty_batch_rejected(self, client):
        r = client.post("/predict/batch", json={"customers": []})
        assert r.status_code == 422

    def test_batch_exceeds_limit(self, client):
        payload = self._make_batch(501)
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 422


from api import _risk_tier


class TestRiskTier:
    @pytest.mark.parametrize(
        "prob, expected",
        [
            (0.10, "Low"),
            (0.29, "Low"),
            (0.30, "Medium"),
            (0.54, "Medium"),
            (0.55, "High"),
            (0.74, "High"),
            (0.75, "Critical"),
            (1.00, "Critical"),
        ],
    )
    def test_tier_boundaries(self, prob, expected):
        assert _risk_tier(prob) == expected