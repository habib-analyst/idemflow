# idemflow — STATUS

**Project:** idemflow — self-diagnosing dead-letter queues (PIVOTED INVENTION)
**Started:** 2026-10-12 ~23:10 PKT
**State:** BUILDING

## What it is
Event workflow engine (webhook/queue listeners, idempotent steps, retry w/ backoff, DLQ routing) where the dead-letter queue **diagnoses why each event failed**: retry-signature cause classification (TRANSIENT / POISON / INFRA) + auto-remediation policy (requeue transients, quarantine poison with human-readable diagnosis, hold infra + alert).

## Kill-gate
- Macro-F1 of signature classifier must beat naive baseline by ≥15pp
- Poison precision ≥ 0.95 (never silently drop real work)
- Else: kill honestly, report negative.

## Progress
- [x] STATUS.md lifeline
- [ ] engine.py (workflow engine)
- [ ] signatures.py (retry-signature features)
- [ ] classifier.py (rules + learned model)
- [ ] remediation.py (policy + diagnosis)
- [ ] simulator.py (synthetic labeled histories)
- [ ] experiments/run_eval.py → results/summary.json
- [ ] tests (green)
- [ ] paper/PAPER.md
- [ ] README.md (branded), LICENSE, pyproject.toml, .gitignore
- [ ] LinkedIn draft
- [ ] push payload staged
