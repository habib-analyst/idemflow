"""Retry-signature feature extraction.

The core idea: the HISTORY of how an event failed is itself a diagnostic
signal. A poison event fails the same way every time, fast. A transient
event fails differently across retries (timeout, then 429, then success-ish
timing). An infra event fails at the same step as everyone else's events.

These features turn that intuition into numbers for the classifier.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

from .engine import (
    CONNECTION, DOWNSTREAM_5XX, LOGIC_ERROR, MISSING_FIELD, RATE_LIMIT,
    SCHEMA_VIOLATION, TIMEOUT, RetryHistory,
)

TRANSIENT_TYPES = {TIMEOUT, RATE_LIMIT, CONNECTION}
POISON_TYPES = {SCHEMA_VIOLATION, MISSING_FIELD, LOGIC_ERROR}
INFRA_TYPES = {DOWNSTREAM_5XX, CONNECTION}

FEATURE_NAMES = [
    "n_attempts",
    "n_distinct_error_types",
    "n_distinct_steps",
    "frac_transient_errors",
    "frac_poison_errors",
    "frac_infra_errors",
    "same_error_every_time",      # 1.0 if all attempts share one error type
    "same_step_every_time",       # 1.0 if all attempts fail at one step
    "mean_latency_s",
    "latency_cv",                 # std/mean: transients vary, poison is fast+stable
    "fast_fail",                  # 1.0 if mean latency < 5ms (deterministic reject)
    "fleet_step_failure_rate",    # cross-event: is this step down for everyone?
    "schema_invalid",             # 1.0 if payload failed schema validation
    "payload_size_log10",
]


def extract_features(hist: RetryHistory) -> Dict[str, float]:
    attempts = hist.attempts
    n = len(attempts)
    err_types = [a.error_type for a in attempts]
    steps = [a.step_name for a in attempts]
    counts = Counter(err_types)

    def frac(types):
        return sum(counts[t] for t in types) / n if n else 0.0

    lats = [a.latency_s for a in attempts]
    mean_lat = sum(lats) / n if n else 0.0
    var = sum((x - mean_lat) ** 2 for x in lats) / n if n else 0.0
    cv = (var ** 0.5) / mean_lat if mean_lat > 0 else 0.0

    import math
    return {
        "n_attempts": float(n),
        "n_distinct_error_types": float(len(counts)),
        "n_distinct_steps": float(len(set(steps))),
        "frac_transient_errors": frac(TRANSIENT_TYPES),
        "frac_poison_errors": frac(POISON_TYPES),
        "frac_infra_errors": frac(INFRA_TYPES),
        "same_error_every_time": 1.0 if len(counts) == 1 and n > 0 else 0.0,
        "same_step_every_time": 1.0 if len(set(steps)) == 1 and n > 0 else 0.0,
        "mean_latency_s": mean_lat,
        "latency_cv": cv,
        "fast_fail": 1.0 if 0 < mean_lat < 0.005 else 0.0,
        "fleet_step_failure_rate": float(hist.fleet_step_failure_rate),
        "schema_invalid": 0.0 if hist.schema_valid else 1.0,
        "payload_size_log10": math.log10(max(hist.payload_size_bytes, 1)),
    }


def feature_vector(hist: RetryHistory) -> List[float]:
    feats = extract_features(hist)
    return [feats[name] for name in FEATURE_NAMES]
