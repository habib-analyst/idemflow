"""Evaluation: signature classifier vs naive baseline on held-out synthetic histories.

Kill-gate:
  1. macro-F1(signature) - macro-F1(naive) >= 0.15
  2. poison precision(signature) >= 0.95
Writes results/summary.json. Exits non-zero with KILL verdict if gate fails.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from idemflow import (  # noqa: E402
    INFRA, POISON, TRANSIENT, NaiveBaseline, RemediationPolicy,
    SignatureClassifier, generate_dataset,
)

N_TRAIN = 240
N_TEST = 300
TRAIN_SEED = 0
TEST_SEED = 1  # held out: never seen in training


def prf(y_true, y_pred, classes):
    out = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": sum(1 for t in y_true if t == c)}
    macro_f1 = sum(v["f1"] for v in out.values()) / len(out)
    return out, macro_f1


def main():
    classes = [TRANSIENT, POISON, INFRA]
    train = generate_dataset(N_TRAIN, TRAIN_SEED)
    test = generate_dataset(N_TEST, TEST_SEED)
    train_h, train_y = zip(*train)
    test_h, test_y = list(zip(*test))
    test_h, test_y = list(test_h), list(test_y)

    naive = NaiveBaseline()
    sig = SignatureClassifier(random_state=0).fit(list(train_h), list(train_y))

    y_naive = naive.predict(test_h)
    y_sig = sig.predict(test_h)

    prf_naive, macro_naive = prf(test_y, y_naive, classes)
    prf_sig, macro_sig = prf(test_y, y_sig, classes)

    # Wasted retries: naive = "retry everything 3 more times".
    naive_wasted = 3 * sum(1 for t in test_y if t in (POISON, INFRA))
    policy = RemediationPolicy()
    decisions = policy.decide_many(test_h, y_sig)
    wr = policy.wasted_retries(decisions, test_y)
    avoided = naive_wasted - wr["wasted"]

    delta_f1 = macro_sig - macro_naive
    poison_prec = prf_sig[POISON]["precision"]
    gate_pass = delta_f1 >= 0.15 and poison_prec >= 0.95

    summary = {
        "n_train": N_TRAIN, "n_test": N_TEST,
        "train_seed": TRAIN_SEED, "test_seed": TEST_SEED,
        "naive": {"per_class": prf_naive, "macro_f1": macro_naive,
                  "wasted_retries": naive_wasted},
        "signature": {"per_class": prf_sig, "macro_f1": macro_sig,
                      "wasted_retries": wr["wasted"],
                      "wasted_retries_avoided": avoided,
                      "feature_importance": sig.feature_importance()},
        "kill_gate": {
            "macro_f1_delta": delta_f1, "delta_bar": 0.15,
            "poison_precision": poison_prec, "poison_precision_bar": 0.95,
            "verdict": "PASS" if gate_pass else "KILL",
        },
    }
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"naive macro-F1:     {macro_naive:.4f}")
    print(f"signature macro-F1: {macro_sig:.4f}  (delta {delta_f1:+.4f}, bar +0.15)")
    print(f"poison precision:   {poison_prec:.4f}  (bar 0.95)")
    for c in classes:
        p = prf_sig[c]
        print(f"  {c:9s} P={p['precision']:.3f} R={p['recall']:.3f} F1={p['f1']:.3f} (n={p['support']})")
    print(f"wasted retries: naive={naive_wasted}, signature={wr['wasted']} "
          f"(avoided {avoided})")
    print("KILL-GATE:", summary["kill_gate"]["verdict"])
    return 0 if gate_pass else 2


if __name__ == "__main__":
    sys.exit(main())
