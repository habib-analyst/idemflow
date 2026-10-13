"""Synthetic retry-history generator with KNOWN ground-truth causes.

Produces RetryHistory objects labeled TRANSIENT / POISON / INFRA, with
realistic retry signatures:

- TRANSIENT: error types vary across retries (timeout, 429, connection);
  latencies vary; sometimes the failing step changes. The event WOULD
  succeed if retried later.
- POISON: the same deterministic error every attempt (schema violation,
  missing field, logic bug); fails fast; payload schema invalid.
- INFRA: downstream 5xx / connection errors at one step, with a HIGH
  fleet_step_failure_rate (everyone else is failing there too).

Seeds control everything; the eval uses held-out seeds never seen in training.
"""
from __future__ import annotations

import random
import time
from typing import List, Tuple

from .engine import (
    CONNECTION, DOWNSTREAM_5XX, LOGIC_ERROR, MISSING_FIELD, RATE_LIMIT,
    SCHEMA_VIOLATION, TIMEOUT, Attempt, RetryHistory,
)

TRANSIENT = "transient"
POISON = "poison"
INFRA = "infra"

_TRANSIENT_ERRORS = [TIMEOUT, RATE_LIMIT, CONNECTION, TIMEOUT, RATE_LIMIT]
_POISON_ERRORS = [SCHEMA_VIOLATION, MISSING_FIELD, LOGIC_ERROR]
_INFRA_ERRORS = [DOWNSTREAM_5XX, DOWNSTREAM_5XX, CONNECTION]

_STEPS = ["validate", "enrich", "charge", "notify"]
_ERROR_MSGS = {
    TIMEOUT: "upstream timed out after 30s",
    RATE_LIMIT: "429 too many requests",
    CONNECTION: "connection reset by peer",
    DOWNSTREAM_5XX: "503 service unavailable",
    SCHEMA_VIOLATION: "schema violation: field 'amount' expected number, got string",
    MISSING_FIELD: "missing required field 'customer_id'",
    LOGIC_ERROR: "KeyError: 'discount_code'",
}


def _attempts(rng: random.Random, step: str, errors: List[str],
              latency_lo: float, latency_hi: float, n: int,
              start_no: int = 1) -> List[Attempt]:
    out = []
    base = time.time()
    for i in range(n):
        et = rng.choice(errors)
        out.append(Attempt(
            attempt_no=start_no + i,
            step_name=step,
            error_type=et,
            error_msg=_ERROR_MSGS[et],
            latency_s=rng.uniform(latency_lo, latency_hi),
            ts=base + i * rng.uniform(1.0, 65.0),
        ))
    return out


def generate_history(rng: random.Random, cause: str, event_id: str) -> RetryHistory:
    """Generate one labeled retry history.

    ~20% of each class is CONFUSABLE: realistic cases where the last error
    type alone misleads (flaky validators, timeout-before-deterministic-
    failure, early-outage low fleet signal). These are the cases that
    motivate signature analysis over last-error heuristics.
    """
    workflow = list(_STEPS)
    confusable = rng.random() < 0.2
    if cause == TRANSIENT:
        # Varies: sometimes the step changes (flaky downstream), errors mix.
        step = rng.choice(_STEPS)
        n = rng.randint(2, 4)
        attempts = _attempts(rng, step, _TRANSIENT_ERRORS, 0.05, 2.5, n)
        if rng.random() < 0.3:  # a second step also flaked once
            other = rng.choice([s for s in _STEPS if s != step])
            attempts += _attempts(rng, other, _TRANSIENT_ERRORS, 0.05, 1.0, 1,
                                  start_no=n + 1)
        if confusable:
            # Flaky validator: last attempt throws a spurious schema error,
            # but the event would succeed on retry (transient root cause).
            attempts += _attempts(rng, step, [SCHEMA_VIOLATION], 0.05, 0.5, 1,
                                  start_no=len(attempts) + 1)
        schema_valid = rng.random() < 0.95
        fleet_rate = rng.uniform(0.0, 0.25)
        size = rng.randint(200, 5000)
    elif cause == POISON:
        step = rng.choice(_STEPS)
        err = rng.choice(_POISON_ERRORS)
        n = rng.randint(2, 4)
        attempts = _attempts(rng, step, [err], 0.0005, 0.004, n)  # fast, deterministic
        if confusable:
            # Timeout on the first attempt, then deterministic failure.
            # Last-error heuristic sees a poison error anyway; the trap is
            # for heuristics that see "timeout first -> transient".
            attempts = (_attempts(rng, step, [TIMEOUT], 1.5, 3.0, 1)
                        + _attempts(rng, step, [err], 0.0005, 0.004, n,
                                    start_no=2))
        schema_valid = err != SCHEMA_VIOLATION or rng.random() < 0.5
        if err == SCHEMA_VIOLATION:
            schema_valid = False
        fleet_rate = rng.uniform(0.0, 0.1)
        size = rng.randint(50, 8000)
    else:  # INFRA
        step = rng.choice(["charge", "notify"])  # downstream-facing steps
        n = rng.randint(2, 4)
        attempts = _attempts(rng, step, _INFRA_ERRORS, 0.5, 5.0, n)
        schema_valid = True
        fleet_rate = rng.uniform(0.6, 1.0)  # everyone is failing here
        if confusable:
            # Early outage: fleet hasn't caught up yet; errors look like
            # ordinary connection flakiness.
            attempts = _attempts(rng, step, [CONNECTION, CONNECTION], 0.5, 2.0, 2)
            fleet_rate = rng.uniform(0.15, 0.35)
        size = rng.randint(200, 5000)
    return RetryHistory(
        event_id=event_id,
        workflow=workflow,
        attempts=attempts,
        payload_size_bytes=size,
        payload_field_count=rng.randint(3, 20),
        schema_valid=schema_valid,
        dead_lettered=True,
        fleet_step_failure_rate=fleet_rate,
    )


def generate_dataset(n: int, seed: int,
                     class_probs: Tuple[float, float, float] = (0.4, 0.4, 0.2)
                     ) -> List[Tuple[RetryHistory, str]]:
    """Generate n labeled (history, cause) pairs. Deterministic in seed."""
    rng = random.Random(seed)
    causes = [TRANSIENT, POISON, INFRA]
    out = []
    for i in range(n):
        cause = rng.choices(causes, weights=class_probs)[0]
        out.append((generate_history(rng, cause, f"evt-{seed}-{i:04d}"), cause))
    return out
