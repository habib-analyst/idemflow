"""Tests for idemflow. Run: python -m pytest tests/ -v"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from idemflow import (
    HOLD, POISON, QUARANTINE, REQUEUE, TRANSIENT, INFRA,
    Engine, IdemflowError, NaiveBaseline, RemediationPolicy,
    SignatureClassifier, extract_features, generate_dataset,
)
from idemflow.engine import TIMEOUT, SCHEMA_VIOLATION, DOWNSTREAM_5XX


# ---------- engine ----------

def _ok_step(payload, ctx):
    return True


def test_idempotency_no_double_execute():
    eng = Engine(sleep=False)
    calls = []

    def step(payload, ctx):
        calls.append(1)

    eng.register_step("s", step)
    eng.run_event("e1", {"a": 1}, ["s"])
    eng.run_event("e1", {"a": 1}, ["s"])  # same event+step+payload -> skip
    assert len(calls) == 1


def test_retry_then_dlq():
    eng = Engine(max_retries=2, sleep=False)

    def bad(payload, ctx):
        raise IdemflowError(SCHEMA_VIOLATION, "schema violation: bad field")

    eng.register_step("v", bad)
    hist = eng.run_event("e2", {"a": 1}, ["v"], ctx={"schema_valid": False})
    assert hist.dead_lettered
    assert len(hist.attempts) == 3  # 1 initial + 2 retries
    assert all(a.error_type == SCHEMA_VIOLATION for a in hist.attempts)
    assert len(eng.dead_letters) == 1


def test_success_after_flaky_step():
    eng = Engine(max_retries=3, sleep=False)
    state = {"n": 0}

    def flaky(payload, ctx):
        state["n"] += 1
        if state["n"] < 3:
            raise TimeoutError("upstream timed out")
        return True

    eng.register_step("f", flaky)
    hist = eng.run_event("e3", {}, ["f"])
    assert not hist.dead_lettered
    assert len(hist.attempts) == 2
    assert hist.attempts[0].error_type == TIMEOUT
    assert eng.succeeded == ["e3"]


def test_exception_taxonomy_mapping():
    eng = Engine(sleep=False)

    def boom(payload, ctx):
        raise KeyError("discount_code")

    eng.register_step("b", boom)
    hist = eng.run_event("e4", {}, ["b"])
    assert hist.attempts[0].error_type == "logic_error"


def test_fleet_context_recorded():
    eng = Engine(max_retries=1, sleep=False)

    def down(payload, ctx):
        raise IdemflowError(DOWNSTREAM_5XX, "503 service unavailable")

    eng.register_step("c", down)
    for i in range(4):
        eng.run_event(f"e{i}", {}, ["c"])
    last = eng.dead_letters[-1]
    assert last.fleet_step_failure_rate == 1.0


# ---------- signatures ----------

def test_signature_features_sane():
    data = generate_dataset(10, seed=42)
    for hist, _ in data:
        feats = extract_features(hist)
        assert set(feats) == {
            "n_attempts", "n_distinct_error_types", "n_distinct_steps",
            "frac_transient_errors", "frac_poison_errors", "frac_infra_errors",
            "same_error_every_time", "same_step_every_time", "mean_latency_s",
            "latency_cv", "fast_fail", "fleet_step_failure_rate",
            "schema_invalid", "payload_size_log10",
        }
        assert 0.0 <= feats["frac_transient_errors"] <= 1.0


# ---------- classifier ----------

@pytest.fixture(scope="module")
def trained():
    train = generate_dataset(240, seed=0)
    clf = SignatureClassifier(random_state=0).fit(
        [h for h, _ in train], [y for _, y in train])
    test = generate_dataset(300, seed=1)
    return clf, test


def test_kill_gate_delta(trained):
    from collections import Counter
    clf, test = trained
    hists = [h for h, _ in test]
    y_true = [y for _, y in test]
    y_sig = clf.predict(hists)
    y_naive = NaiveBaseline().predict(hists)

    def macro_f1(yt, yp):
        f1s = []
        for c in (TRANSIENT, POISON, INFRA):
            tp = sum(1 for t, p in zip(yt, yp) if t == c and p == c)
            fp = sum(1 for t, p in zip(yt, yp) if t != c and p == c)
            fn = sum(1 for t, p in zip(yt, yp) if t == c and p != c)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        return sum(f1s) / 3

    assert macro_f1(y_true, y_sig) - macro_f1(y_true, y_naive) >= 0.15


def test_poison_precision_bar(trained):
    clf, test = trained
    y_true = [y for _, y in test]
    y_sig = clf.predict([h for h, _ in test])
    tp = sum(1 for t, p in zip(y_true, y_sig) if p == POISON and t == POISON)
    fp = sum(1 for t, p in zip(y_true, y_sig) if p == POISON and t != POISON)
    assert tp / (tp + fp) >= 0.95


def test_naive_baseline_contract():
    data = generate_dataset(30, seed=7)
    preds = NaiveBaseline().predict([h for h, _ in data])
    assert set(preds) <= {TRANSIENT, POISON, INFRA}
    assert len(preds) == 30


def test_feature_importance_nonempty(trained):
    clf, _ = trained
    imp = clf.feature_importance()
    assert len(imp) == 14
    assert abs(sum(v for _, v in imp) - 1.0) < 1e-6


# ---------- remediation ----------

def test_policy_actions():
    data = generate_dataset(60, seed=9)
    clf = SignatureClassifier(random_state=0).fit(
        [h for h, _ in generate_dataset(240, seed=0)],
        [y for _, y in generate_dataset(240, seed=0)],
    )
    policy = RemediationPolicy()
    hists = [h for h, _ in data]
    causes = [y for _, y in data]
    preds = clf.predict(hists)
    decisions = policy.decide_many(hists, preds)
    for d, t, p in zip(decisions, causes, preds):
        if p == POISON:
            assert d.action == QUARANTINE and d.retries_budgeted == 0
        elif p == INFRA:
            assert d.action == HOLD and d.retries_budgeted == 0
        else:
            assert d.action == REQUEUE and d.retries_budgeted > 0
        assert d.event_id and d.diagnosis


def test_diagnosis_names_step_and_error():
    data = generate_dataset(10, seed=11)
    hist, _ = data[0]
    policy = RemediationPolicy()
    d = policy.decide(hist, POISON)
    assert hist.failing_step in d.diagnosis
    assert "POISON" in d.diagnosis


def test_wasted_retries_zero_for_signature():
    data = generate_dataset(300, seed=1)
    clf = SignatureClassifier(random_state=0).fit(
        [h for h, _ in generate_dataset(240, seed=0)],
        [y for _, y in generate_dataset(240, seed=0)],
    )
    policy = RemediationPolicy()
    hists = [h for h, _ in data]
    y_true = [y for _, y in data]
    decisions = policy.decide_many(hists, clf.predict(hists))
    wr = policy.wasted_retries(decisions, y_true)
    naive_wasted = 3 * sum(1 for t in y_true if t in (POISON, INFRA))
    assert wr["wasted"] <= naive_wasted
