"""
04_eval.py

Same eval methodology as the ticket triage project: per-class precision/
recall/F1 (not just overall accuracy), and an explicit check for collapse
to the majority class. Compares baseline vs. context-engineered predictions
if both are present.

Tested against real severity labels using synthetic prediction files
(a majority-class-collapse case and a genuinely mixed case) to confirm
the collapse check actually fires when it should and stays quiet when
predictions are reasonably distributed.
"""
import argparse
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

PROC_DIR = Path("data/processed")


def evaluate(df: pd.DataFrame, name: str):
    y_true = df["true_label"]
    y_pred = df["pred_label"]

    unparsed = (y_pred == -1).sum()
    valid = df[df["pred_label"] != -1]

    precision, recall, f1, support = precision_recall_fscore_support(
        valid["true_label"], valid["pred_label"], labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(valid["true_label"], valid["pred_label"], labels=[0, 1])
    pred_dist = valid["pred_label"].value_counts(normalize=True).to_dict()

    print(f"\n=== {name} ===")
    print(f"n={len(df)} (unparsed responses: {unparsed})")
    print(f"Predicted label distribution: {pred_dist}")
    print(f"Confusion matrix [rows=true, cols=pred] 0/1:\n{cm}")
    for label, p, r, f, s in zip([0, 1], precision, recall, f1, support):
        print(f"  class {label}: precision={p:.3f} recall={r:.3f} f1={f:.3f} support={s}")

    # collapse check: did the model just predict the majority class every time?
    majority_frac = max(pred_dist.values()) if pred_dist else 0
    collapsed = majority_frac > 0.97 or (len(pred_dist) == 1)
    print(f"COLLAPSE CHECK: {'*** COLLAPSED to majority class ***' if collapsed else 'not collapsed'} "
          f"(top predicted class = {majority_frac:.1%} of predictions)")
    return {"name": name, "f1": f1, "collapsed": collapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sev")
    args = parser.parse_args()

    baseline_path = PROC_DIR / f"{args.task}_baseline_predictions.csv"
    context_path = PROC_DIR / f"{args.task}_context_predictions.csv"

    results = []
    if baseline_path.exists():
        results.append(evaluate(pd.read_csv(baseline_path), "BASELINE (no retrieval)"))
    if context_path.exists():
        results.append(evaluate(pd.read_csv(context_path), "CONTEXT-ENGINEERED (with retrieval)"))

    if len(results) == 2:
        print("\n=== COMPARISON ===")
        b, c = results
        print(f"Baseline class-1 (severe) F1: {b['f1'][1]:.3f}")
        print(f"Context  class-1 (severe) F1: {c['f1'][1]:.3f}")
        if b["collapsed"] and not c["collapsed"]:
            print("Retrieval context appears to have fixed a majority-class collapse.")
        elif b["collapsed"] and c["collapsed"]:
            print("Retrieval context did NOT fix the collapse — still defaulting to majority class.")
        elif not b["collapsed"] and not c["collapsed"]:
            print("Neither run collapsed; compare F1 to see if context helped at the margin.")


if __name__ == "__main__":
    main()
