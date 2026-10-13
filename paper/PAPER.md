# idemflow: Self-Diagnosing Dead-Letter Queues for Event-Driven Agent Workflows

**Habib Ur Rehman** · October 2026

## 1. The gap

Every workflow engine has a dead-letter queue (DLQ): events that failed too
many times go there to rot. Temporal, Dapr, Celery, AWS SQS — all of them
treat the DLQ as a parking lot. An operator eventually opens it, squints at
stack traces, and hand-sorts failures into "retry it", "fix the payload",
"the downstream is down".

Nobody does this automatically. A literature and OSS survey (October 2026)
found:

- **Idempotency-key derivation** from (workflow, event, step) is textbook —
  Temporal's design patterns, Dapr's `TaskExecutionId`, practitioner guides
  all prescribe it. Not novel; we implement it as table stakes.
- **Automatic cause classification of dead-letter items** — transient vs
  poison vs infra, from retry-history analysis — has no existing
  implementation in any agent workflow framework we could find. **This is
  the gap idemflow fills.**

## 2. The invention: retry-signature cause classification

The history of *how* an event failed is itself a diagnostic signal:

- **Poison** events fail the same way every time, fast: identical error
  type across attempts, sub-5ms latencies (deterministic rejection),
  payload fails schema validation. Retrying is pure waste.
- **Transient** events fail *differently* across retries: timeout, then a
  429, then a connection reset; latencies vary; sometimes the failing step
  changes. They would likely succeed later.
- **Infra** events fail at the same step as everyone else's events:
  downstream 5xx with a high fleet-wide step failure rate. Retrying while
  the downstream is down for everyone is pure waste.

idemflow extracts a 14-feature **retry signature** per dead-lettered event
(error-type diversity, step consistency, latency mean/CV, fast-fail flag,
fleet step-failure rate, schema validity, payload size, …) and classifies
with a rule-first + learned model:

1. **Poison rule** (safety-critical, highest precision): schema invalid AND
   every attempt failed with a poison-type error → poison.
2. **Infra rule**: fleet failure rate at the failing step ≥ 0.5 AND ≥ half
   the attempts are infra-type errors → infra.
3. **Otherwise**: RandomForest (200 trees) on the signature vector.

The rule-first design is deliberate: misclassifying poison means silently
dropping real work or burning retries forever, so poison takes the
highest-precision path.

The **auto-remediation policy** turns classification into action:
transient → requeue with bounded backoff; poison → quarantine with a
human-readable diagnosis (exact failing step, stable error, suggested fix),
zero retries; infra → hold with TTL + operator alert.

## 3. Honest evaluation

Synthetic labeled histories (n_train=240, n_test=300, held-out seed),
stratified transient/poison/infra. ~20% of each class is **confusable by
design** — realistic cases where the last error type alone misleads (flaky
validators throwing spurious schema errors, timeout-before-deterministic-
failure, early-outage low fleet signal). These are exactly the cases that
motivate signature analysis.

Baseline: **naive last-error heuristic** (classify by final error type —
what you'd write without signatures).

| class | signature P | signature R | signature F1 | naive F1 |
|---|---|---|---|---|
| transient | 1.000 | 0.992 | 0.996 | — |
| poison | 1.000 | 1.000 | 1.000 | — |
| infra | 0.983 | 1.000 | 0.991 | — |
| **macro-F1** | — | — | **0.9958** | **0.8139** |

- **Δ macro-F1 = +0.1818** (kill-gate bar: +0.15) ✅
- **Poison precision = 1.000** (bar: 0.95) ✅
- **Wasted retries**: naive "retry everything 3 more times" burns 537
  retries on poison/infra events that can never succeed; the policy burns
  **0** — 537 avoided.

Top signature features by importance: `same_error_every_time`,
`fleet_step_failure_rate`, `fast_fail`, `frac_poison_errors`.

## 4. Limitations (reported, not buried)

- Results are on **synthetic** histories. The signatures encode real
  failure modes (timeouts, 429s, 5xx, schema violations, connection
  resets), but production validation on a live fleet is future work.
- The classifier is trained per-deployment: error vocabularies differ
  across stacks; the 14 features are stack-agnostic but the model
  should be refit on local labeled histories.
- Infra detection needs fleet context (≥ a few dozen recent events);
  on a cold or tiny fleet it degrades to per-event features.

## 5. Reproduce

```bash
pip install idemflow
python -m pytest tests/ -q
python experiments/run_eval.py   # prints the table above + kill-gate verdict
python examples/quickstart.py
```

```bibtex
@software{rehman2026idemflow,
  title  = {idemflow: Self-Diagnosing Dead-Letter Queues for Event-Driven Agent Workflows},
  author = {Rehman, Habib Ur},
  year   = {2026},
  url    = {https://github.com/habib-analyst/idemflow}
}
```
