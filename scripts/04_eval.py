"""
04_eval.py

Evaluate baseline vs. context-engineered classification against the real
majority-class baselines in this dataset:

  severity:  77.2% of test bugs are "not severe"
  fix-time:  79.4% of test bugs are "fast fix"

Any reported accuracy near those numbers, on either condition, is a
warning sign that the model is defaulting to the majority class rather
than genuinely discriminating; the exact failure mode the ticket triage
project's eval harness caught.
"""

import json
import argparse
from pathlib import Path
from sklearn.metrics import classification_report

RESULTS_DIR = Path("./results")

MAJORITY_BASELINES = {
    "sev": 0.7723,
    "fix": 0.7935,
}


def load_results(task: str) -> list:
    with open(RESULTS_DIR / f"{task}_classification_results.json") as f:
        return json.load(f)


def evaluate_condition(results: list, condition: str, majority_baseline: float) -> dict:
    subset = [r for r in results if r["condition"] == condition and not r["parse_failed"]]
    if not subset:
        print(f"  No valid (non-parse-failed) results for {condition}")
        return {}

    y_true = [r["true_label"] for r in subset]
    y_pred = [r["prediction"] for r in subset]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = report["accuracy"]

    print(f"\n{'='*60}")
    print(f"{condition.upper()}")
    print(f"{'='*60}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Majority-class baseline for this task: {majority_baseline:.1%}")

    diff = accuracy - majority_baseline
    if diff <= 0.02:
        print(f"  WARNING: accuracy is within 2 points of the majority baseline.")
        print(f"  This model may not be discriminating at all, check per-class recall below.")
    else:
        print(f"  Beats majority baseline by {diff:.1%}, a genuine signal worth trusting more.")

    print(f"\n  Per-class recall:")
    for label, metrics in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        recall = metrics["recall"]
        flag = " <- never predicted, collapsed" if recall == 0.0 else \
               " <- WARNING: may be defaulting here" if recall > 0.95 else ""
        print(f"    {label}: recall = {recall:.3f}{flag}")

    parse_fail_rate = sum(1 for r in results if r["condition"] == condition and r["parse_failed"]) / \
        len([r for r in results if r["condition"] == condition])
    mean_latency = sum(r["latency_ms"] for r in subset) / len(subset)
    print(f"\n  Parse failure rate: {parse_fail_rate:.1%}")
    print(f"  Mean latency: {mean_latency:.0f}ms")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sev", "fix"], default="sev")
    args = parser.parse_args()

    results = load_results(args.task)
    majority_baseline = MAJORITY_BASELINES[args.task]

    baseline_report = evaluate_condition(results, "baseline", majority_baseline)
    context_report = evaluate_condition(results, "context_engineered", majority_baseline)

    print(f"\n{'='*60}")
    print("HONEST VERDICT")
    print(f"{'='*60}")
    if baseline_report and context_report:
        base_acc = baseline_report.get("accuracy", 0)
        ctx_acc = context_report.get("accuracy", 0)
        print(f"Baseline accuracy: {base_acc:.1%} (majority baseline: {majority_baseline:.1%})")
        print(f"Context-engineered accuracy: {ctx_acc:.1%}")
        print(f"Change from context: {ctx_acc - base_acc:+.1%}")
        print("\nCheck the per-class recall above for both conditions, not just this")
        print("summary. A model can improve overall accuracy while still collapsing")
        print("on one or more classes, the exact pattern that mattered last time.")

    with open(RESULTS_DIR / f"{args.task}_eval_summary.json", "w") as f:
        json.dump({"baseline": baseline_report, "context_engineered": context_report}, f, indent=2)
    print(f"\nFull reports saved to {RESULTS_DIR}/{args.task}_eval_summary.json")


if __name__ == "__main__":
    main()
