"""idemflow quickstart: run a flaky workflow, watch the dead-letter queue
diagnose each failure and auto-remediate.

Expected output: 3 events dead-lettered; the poison event is quarantined
with a diagnosis, the transient is requeued, the infra event is held.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from idemflow import (
    DOWNSTREAM_5XX, Engine, IdemflowError, NaiveBaseline, RemediationPolicy,
    SCHEMA_VIOLATION, SignatureClassifier, TIMEOUT, generate_dataset,
)

# --- 1. a tiny workflow engine -------------------------------------------
eng = Engine(max_retries=2, sleep=False)

calls = {"charge": 0}


def validate(payload, ctx):
    if not isinstance(payload.get("amount"), (int, float)):
        raise IdemflowError(SCHEMA_VIOLATION, "schema violation: 'amount' must be a number")


def charge(payload, ctx):
    calls["charge"] += 1
    if ctx.get("downstream_down"):
        raise IdemflowError(DOWNSTREAM_5XX, "503 service unavailable")
    if calls["charge"] % 2 == 1:
        raise TimeoutError("upstream timed out")


eng.register_step("validate", validate)
eng.register_step("charge", charge)

# poison: bad payload, deterministic failure
h1 = eng.run_event("evt-poison", {"amount": "lots"}, ["validate", "charge"],
                   ctx={"schema_valid": False})
# transient: flaky charge step (fails once, recovers on retry — never reaches DLQ)
h2 = eng.run_event("evt-flaky", {"amount": 42}, ["validate", "charge"])
print(f"flaky event recovered: {'evt-flaky' in eng.succeeded} (expected True)")
# infra: downstream down for everyone
h3 = eng.run_event("evt-infra", {"amount": 7}, ["validate", "charge"],
                   ctx={"downstream_down": True})

print(f"dead-lettered: {len(eng.dead_letters)} (expected 2: "
      f"the flaky event recovered on retry, as designed)")

# --- 2. train the diagnoser on synthetic histories ------------------------
train = generate_dataset(240, seed=0)
clf = SignatureClassifier(random_state=0).fit(
    [h for h, _ in train], [y for _, y in train])

# --- 3. diagnose + remediate ------------------------------------------------
policy = RemediationPolicy()
for hist in eng.dead_letters:
    cause = clf.predict_one(hist)
    d = policy.decide(hist, cause)
    print(f"\n{hist.event_id}: predicted={cause} action={d.action}")
    print(d.diagnosis)
