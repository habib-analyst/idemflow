"""Cause classification for dead-lettered events.

Two classifiers:

1. NaiveBaseline — what you'd do without retry signatures: classify by the
   LAST error type alone (timeout/429/conn -> transient, schema/missing/logic
   -> poison, 5xx -> infra, unknown -> transient). Honest, weak, and the
   kill-gate baseline.

2. SignatureClassifier — high-precision deterministic rules for the
   safety-critical cases, then a learned model on the full retry-signature
   feature vector for the rest:
     - POISON rule: payload schema invalid AND every attempt failed with a
       poison-type error  -> poison (deterministic, never succeeds on retry)
     - INFRA rule: fleet failure rate at the failing step >= 0.5 AND at
       least half the attempts are infra-type errors -> infra
     - otherwise: RandomForest on the 14 signature features.

The rule-first design is deliberate: poison misclassification means silently
dropping real work or burning retries forever, so poison gets the
highest-precision path.
"""
from __future__ import annotations

from typing import List, Tuple

from .engine import (
    CONNECTION, DOWNSTREAM_5XX, LOGIC_ERROR, MISSING_FIELD, RATE_LIMIT,
    SCHEMA_VIOLATION, TIMEOUT, RetryHistory,
)
from .signatures import FEATURE_NAMES, extract_features, feature_vector
from .simulator import INFRA, POISON, TRANSIENT

_TRANSIENT_TYPES = {TIMEOUT, RATE_LIMIT, CONNECTION}
_POISON_TYPES = {SCHEMA_VIOLATION, MISSING_FIELD, LOGIC_ERROR}
_INFRA_TYPES = {DOWNSTREAM_5XX, CONNECTION}

CLASSES = [TRANSIENT, POISON, INFRA]


class NaiveBaseline:
    """Classify by the last error type only. No signature analysis."""

    name = "naive-last-error"

    def predict_one(self, hist: RetryHistory) -> str:
        if not hist.attempts:
            return TRANSIENT
        last = hist.attempts[-1].error_type
        if last in _POISON_TYPES:
            return POISON
        if last == DOWNSTREAM_5XX:
            return INFRA
        return TRANSIENT  # timeouts, 429s, connections, unknowns -> retry

    def predict(self, hists: List[RetryHistory]) -> List[str]:
        return [self.predict_one(h) for h in hists]


class SignatureClassifier:
    """Rule-first + learned signature classifier.

    fit() trains the RandomForest on labeled histories. predict_one()
    applies the safety rules first, then the model.
    """

    name = "signature-rf"

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.model = None

    # -- rules -----------------------------------------------------------
    @staticmethod
    def _rule_poison(hist: RetryHistory) -> bool:
        if not hist.attempts:
            return False
        return (not hist.schema_valid
                and all(a.error_type in _POISON_TYPES for a in hist.attempts))

    @staticmethod
    def _rule_infra(hist: RetryHistory) -> bool:
        if not hist.attempts:
            return False
        n = len(hist.attempts)
        frac_infra = sum(1 for a in hist.attempts
                         if a.error_type in _INFRA_TYPES) / n
        return hist.fleet_step_failure_rate >= 0.5 and frac_infra >= 0.5

    # -- learning ---------------------------------------------------------
    def fit(self, hists: List[RetryHistory], labels: List[str]) -> "SignatureClassifier":
        from sklearn.ensemble import RandomForestClassifier
        X = [feature_vector(h) for h in hists]
        self.model = RandomForestClassifier(
            n_estimators=200, random_state=self.random_state,
            min_samples_leaf=3, n_jobs=1,
        )
        self.model.fit(X, labels)
        return self

    def predict_one(self, hist: RetryHistory) -> str:
        if self._rule_poison(hist):
            return POISON
        if self._rule_infra(hist):
            return INFRA
        if self.model is None:
            return NaiveBaseline().predict_one(hist)
        return self.model.predict([feature_vector(hist)])[0]

    def predict(self, hists: List[RetryHistory]) -> List[str]:
        return [self.predict_one(h) for h in hists]

    def feature_importance(self) -> List[Tuple[str, float]]:
        if self.model is None:
            return []
        imps = self.model.feature_importances_
        return sorted(zip(FEATURE_NAMES, imps), key=lambda t: -t[1])
