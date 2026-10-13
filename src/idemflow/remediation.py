"""Auto-remediation policy + human-readable diagnosis.

The policy turns a classification into an action:

- TRANSIENT -> requeue with exponential backoff (bounded requeues; the event
  would likely succeed later, so we spend retries here).
- POISON   -> quarantine + diagnosis. NEVER retried: retrying a poison event
  is pure waste. The diagnosis names the exact failing step, the stable
  error signature, and a suggested fix.
- INFRA    -> hold with a TTL + operator alert. Retrying while the
  downstream is down for everyone is pure waste; resume when healthy.

Diagnosis is template-generated from the retry signature — no LLM needed,
deterministic, and auditable.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .engine import RetryHistory
from .simulator import INFRA, POISON, TRANSIENT

REQUEUE = "requeue"
QUARANTINE = "quarantine"
HOLD = "hold"


@dataclass
class RemediationDecision:
    event_id: str
    predicted_cause: str
    action: str
    backoff_s: Optional[float] = None
    hold_ttl_s: Optional[float] = None
    diagnosis: str = ""
    retries_budgeted: int = 0  # extra retries this decision spends


@dataclass
class PolicyConfig:
    transient_requeues: int = 2
    transient_backoff_base_s: float = 30.0
    infra_hold_ttl_s: float = 900.0


class RemediationPolicy:
    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self.alerts: List[str] = []  # operator alerts (infra holds)

    def diagnose(self, hist: RetryHistory, predicted_cause: str) -> str:
        step = hist.failing_step or "?"
        err_types = [a.error_type for a in hist.attempts]
        top_err, top_n = Counter(err_types).most_common(1)[0] if err_types else ("?", 0)
        lines = [
            f"event {hist.event_id}: dead-lettered after {len(hist.attempts)} attempts, "
            f"all at step '{step}'.",
            f"stable error: '{top_err}' ({top_n}/{len(hist.attempts)} attempts).",
        ]
        if predicted_cause == POISON:
            lines.append("verdict: POISON — deterministic failure, will never succeed on retry.")
            if not hist.schema_valid:
                lines.append("payload failed schema validation: fix the producer or update the schema.")
            else:
                lines.append(f"suggested fix: inspect step '{step}' for the '{top_err}' "
                             "condition (bad field value or code bug); correct the payload or the step.")
            lines.append("action: quarantined, zero further retries budgeted.")
        elif predicted_cause == INFRA:
            lines.append(f"verdict: INFRA — step '{step}' is failing for "
                         f"{hist.fleet_step_failure_rate:.0%} of recent events fleet-wide.")
            lines.append("suggested fix: check downstream health for "
                         f"'{step}'; resume held events when healthy.")
            lines.append(f"action: held for {self.config.infra_hold_ttl_s:.0f}s, operator alerted.")
        else:
            lines.append("verdict: TRANSIENT — error signature varies across retries; "
                         "likely to succeed later.")
            lines.append(f"action: requeued with backoff "
                         f"({self.config.transient_requeues} retries budgeted).")
        return "\n".join(lines)

    def decide(self, hist: RetryHistory, predicted_cause: str) -> RemediationDecision:
        if predicted_cause == POISON:
            return RemediationDecision(
                event_id=hist.event_id, predicted_cause=predicted_cause,
                action=QUARANTINE, retries_budgeted=0,
                diagnosis=self.diagnose(hist, predicted_cause),
            )
        if predicted_cause == INFRA:
            self.alerts.append(
                f"INFRA HOLD: step '{hist.failing_step}' failing for "
                f"{hist.fleet_step_failure_rate:.0%} of fleet; "
                f"event {hist.event_id} held {self.config.infra_hold_ttl_s:.0f}s"
            )
            return RemediationDecision(
                event_id=hist.event_id, predicted_cause=predicted_cause,
                action=HOLD, hold_ttl_s=self.config.infra_hold_ttl_s,
                retries_budgeted=0,
                diagnosis=self.diagnose(hist, predicted_cause),
            )
        # TRANSIENT
        return RemediationDecision(
            event_id=hist.event_id, predicted_cause=predicted_cause,
            action=REQUEUE, backoff_s=self.config.transient_backoff_base_s,
            retries_budgeted=self.config.transient_requeues,
            diagnosis=self.diagnose(hist, predicted_cause),
        )

    def decide_many(self, hists: List[RetryHistory],
                    causes: List[str]) -> List[RemediationDecision]:
        return [self.decide(h, c) for h, c in zip(hists, causes)]

    def wasted_retries(self, decisions: List[RemediationDecision],
                       true_causes: List[str]) -> Dict[str, int]:
        """Retries budgeted on events that could never succeed (poison/infra)."""
        wasted = sum(d.retries_budgeted for d, t in zip(decisions, true_causes)
                     if t in (POISON, INFRA))
        total = sum(d.retries_budgeted for d in decisions)
        return {"wasted": wasted, "total_budgeted": total}
