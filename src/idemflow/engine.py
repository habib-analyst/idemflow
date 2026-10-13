"""idemflow engine: event-driven workflow execution with idempotent steps,
retry with exponential backoff, and dead-letter routing with full retry history.

The engine is the table-stakes half of idemflow. The invention lives in
signatures.py / classifier.py / remediation.py: the dead-letter queue
diagnoses WHY each event failed and auto-remediates.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

# Canonical error taxonomy. Steps raise real exceptions; the engine maps them
# to these types so the classifier works on a stable vocabulary.
TIMEOUT = "timeout"
RATE_LIMIT = "rate_limit"
CONNECTION = "connection"
DOWNSTREAM_5XX = "downstream_5xx"
SCHEMA_VIOLATION = "schema_violation"
MISSING_FIELD = "missing_field"
LOGIC_ERROR = "logic_error"
PAYLOAD_TOO_LARGE = "payload_too_large"
UNKNOWN = "unknown"

_EXCEPTION_MAP = [
    (TimeoutError, TIMEOUT),
    (ConnectionError, CONNECTION),
]


def classify_exception(exc: BaseException) -> str:
    """Map a raised exception to the canonical error type."""
    # Explicit marker attribute wins (steps can raise IdemflowError("schema_violation", ...))
    et = getattr(exc, "idemflow_error_type", None)
    if et:
        return et
    for cls, name in _EXCEPTION_MAP:
        if isinstance(exc, cls):
            return name
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return RATE_LIMIT
    if "503" in msg or "502" in msg or "504" in msg or "service unavailable" in msg:
        return DOWNSTREAM_5XX
    if "schema" in msg or "validation" in msg:
        return SCHEMA_VIOLATION
    if "missing" in msg and "field" in msg:
        return MISSING_FIELD
    return LOGIC_ERROR if isinstance(exc, (KeyError, ValueError, TypeError, AttributeError)) else UNKNOWN


class IdemflowError(Exception):
    """Raise from a step to declare an explicit canonical error type."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.idemflow_error_type = error_type


@dataclass
class Attempt:
    attempt_no: int
    step_name: str
    error_type: str
    error_msg: str
    latency_s: float
    ts: float


@dataclass
class RetryHistory:
    event_id: str
    workflow: List[str]
    attempts: List[Attempt] = field(default_factory=list)
    payload_size_bytes: int = 0
    payload_field_count: int = 0
    schema_valid: bool = True
    dead_lettered: bool = False
    # Cross-event context, filled by the engine at DLQ time:
    fleet_step_failure_rate: float = 0.0  # fraction of recent events failing at the failing step

    @property
    def failing_step(self) -> Optional[str]:
        return self.attempts[-1].step_name if self.attempts else None

    @property
    def error_types(self) -> List[str]:
        return [a.error_type for a in self.attempts]


@dataclass
class StepResult:
    ok: bool
    value: Any = None


class Engine:
    """Minimal event workflow engine.

    - Steps are named callables: fn(payload, ctx) -> Any (raise on failure).
    - Idempotency: key = sha256(event_id | step_name | payload_hash). A
      completed (event, step) is never re-executed, even across requeues.
    - Retry: per-event max_retries with exponential backoff (+ jitter).
    - DLQ: after max_retries, the event lands in dead_letters with its full
      RetryHistory, including fleet context for infra detection.
    """

    def __init__(self, max_retries: int = 3, backoff_base_s: float = 0.05,
                 fleet_window: int = 50, sleep: bool = True):
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.sleep = sleep
        self.steps: Dict[str, Callable] = {}
        self.completed_keys: set = set()          # idempotency store
        self.dead_letters: List[RetryHistory] = []
        self.succeeded: List[str] = []
        # Fleet context: recent (step_name, failed?) outcomes for INFRA detection
        self._fleet: Deque[tuple] = deque(maxlen=fleet_window)

    def register_step(self, name: str, fn: Callable[[dict, dict], Any]) -> None:
        self.steps[name] = fn

    @staticmethod
    def _payload_hash(payload: dict) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def _idem_key(self, event_id: str, step_name: str, payload: dict) -> str:
        return hashlib.sha256(
            f"{event_id}|{step_name}|{self._payload_hash(payload)}".encode()
        ).hexdigest()

    def _fleet_failure_rate(self, step_name: str) -> float:
        rel = [f for (s, f) in self._fleet if s == step_name]
        return sum(rel) / len(rel) if rel else 0.0

    def run_event(self, event_id: str, payload: dict, workflow: List[str],
                  ctx: Optional[dict] = None) -> RetryHistory:
        ctx = ctx or {}
        hist = RetryHistory(
            event_id=event_id,
            workflow=list(workflow),
            payload_size_bytes=len(json.dumps(payload, default=str)),
            payload_field_count=len(payload) if isinstance(payload, dict) else 0,
            schema_valid=bool(ctx.get("schema_valid", True)),
        )
        for step_name in workflow:
            key = self._idem_key(event_id, step_name, payload)
            if key in self.completed_keys:
                continue  # idempotent skip
            fn = self.steps[step_name]
            attempt_no = 0
            while True:
                attempt_no += 1
                t0 = time.monotonic()
                try:
                    fn(payload, ctx)
                    self.completed_keys.add(key)
                    self._fleet.append((step_name, False))
                    break
                except Exception as exc:  # noqa: BLE001 - engine must survive step bugs
                    latency = time.monotonic() - t0
                    et = classify_exception(exc)
                    hist.attempts.append(Attempt(
                        attempt_no=attempt_no, step_name=step_name,
                        error_type=et, error_msg=str(exc)[:200],
                        latency_s=latency, ts=time.time(),
                    ))
                    self._fleet.append((step_name, True))
                    if attempt_no > self.max_retries:
                        hist.dead_lettered = True
                        hist.fleet_step_failure_rate = self._fleet_failure_rate(step_name)
                        self.dead_letters.append(hist)
                        return hist
                    if self.sleep:
                        delay = self.backoff_base_s * (2 ** (attempt_no - 1))
                        time.sleep(min(delay, 1.0))
        self.succeeded.append(event_id)
        return hist
