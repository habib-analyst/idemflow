[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![GitHub stars](https://img.shields.io/github/stars/habib-analyst/idemflow.svg)](https://github.com/habib-analyst/idemflow/stargazers)

# idemflow — the dead-letter queue that diagnoses itself

Every workflow engine has a dead-letter queue where failed events go to rot. **idemflow is the first one that tells you *why* each event failed** — transient, poison, or infra — from the *signature* of its retry history, then auto-remediates: requeue transients, quarantine poison with a human-readable diagnosis, hold infra events until the downstream recovers.

> 30-second why: your DLQ is a parking lot; idemflow turns it into a triage nurse.

## Quickstart

```bash
pip install idemflow
```

```python
from idemflow import Engine, SignatureClassifier, RemediationPolicy, generate_dataset

eng = Engine(max_retries=2)
eng.register_step("charge", charge_fn)          # raise on failure; engine maps exceptions
hist = eng.run_event("evt-1", {"amount": 42}, ["validate", "charge"])

train = generate_dataset(240, seed=0)           # or fit on your own labeled histories
clf = SignatureClassifier().fit([h for h, _ in train], [y for _, y in train])

policy = RemediationPolicy()
for dead in eng.dead_letters:
    decision = policy.decide(dead, clf.predict_one(dead))
    print(decision.action)      # requeue | quarantine | hold
    print(decision.diagnosis)   # exact step, stable error, suggested fix
```

Run the demo:

```bash
python examples/quickstart.py
```

Expected output (abridged):

```
flaky event recovered: True (expected True)
dead-lettered: 2 (expected 2: the flaky event recovered on retry, as designed)

evt-poison: predicted=poison action=quarantine
event evt-poison: dead-lettered after 3 attempts, all at step 'validate'.
stable error: 'schema_violation' (3/3 attempts).
verdict: POISON — deterministic failure, will never succeed on retry.
payload failed schema validation: fix the producer or update the schema.
action: quarantined, zero further retries budgeted.

evt-infra: predicted=infra action=hold
...
verdict: INFRA — step 'charge' is failing for 80% of recent events fleet-wide.
action: held for 900s, operator alerted.
```

## Measured results

Synthetic labeled histories (n_train=240, n_test=300 held-out seed; ~20% confusable-by-design cases where the last error type alone misleads). Baseline: naive last-error heuristic.

| metric | signature classifier | naive baseline |
|---|---|---|
| macro-F1 | **0.9958** | 0.8139 |
| Δ macro-F1 | **+0.1818** (bar +0.15) | — |
| poison precision | **1.000** (bar 0.95) | — |
| wasted retries on poison/infra | **0** (537 avoided) | 537 |

Per-class: transient F1 0.996 · poison F1 1.000 · infra F1 0.991. 13/13 tests green.

## How it works

1. **Engine** (`engine.py`): idempotent steps (key = `sha256(event|step|payload)`), exponential-backoff retry, DLQ routing with full `RetryHistory` + fleet context.
2. **Signatures** (`signatures.py`): 14-feature retry signature — error-type diversity, step consistency, latency mean/CV, fast-fail flag, fleet step-failure rate, schema validity.
3. **Classifier** (`classifier.py`): high-precision rules for poison/infra first, RandomForest on the signature for the rest.
4. **Remediation** (`remediation.py`): policy → requeue / quarantine + diagnosis / hold + alert. Deterministic, auditable, no LLM needed.

Full method, gap analysis, and limitations: [`paper/PAPER.md`](paper/PAPER.md).

## Citation

```bibtex
@software{rehman2026idemflow,
  title  = {idemflow: Self-Diagnosing Dead-Letter Queues for Event-Driven Agent Workflows},
  author = {Rehman, Habib Ur},
  year   = {2026},
  url    = {https://github.com/habib-analyst/idemflow}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
