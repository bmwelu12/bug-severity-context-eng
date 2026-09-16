"""
05_export_docs_json.py

Exports full per-row results (not just aggregate scores) for the interactive
explorer at docs/index.html: every test bug's description, true label,
baseline prediction, context-engineered prediction, and the actual
retrieved neighbors used for the context prediction.

Re-derives the exact same 300-row sample 03_classify.py used (same CSV,
same random_state=42) and re-runs retrieval (deterministic, no API calls)
to recover which neighbors were shown to the model for each context
prediction — that data was never saved by 03_classify.py itself. Cross-checks
every row's true_label against data/raw/results/sev_classification_results.json
before trusting the alignment, and aborts rather than exporting silently
wrong data if anything doesn't line up.

Run from data/raw/ (same convention as the other scripts):
    cd data/raw
    python3 ../../scripts/05_export_docs_json.py --task sev --sample_size 300
"""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

RESULTS_DIR = Path("results")
ARTIFACT_DIR = Path("artifacts")
DOCS_DIR = Path("../../docs")

LABEL_NAMES = {0: "not severe", 1: "severe"}


def load_retrieval_module():
    spec = importlib.util.spec_from_file_location("retrieval_mod", "../../scripts/02_retrieval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_metrics(rows, condition_key):
    y_true = [r["trueLabel"] for r in rows]
    y_pred = [r[condition_key]["prediction"] for r in rows]
    labels = ["not severe", "severe"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    pred_dist = {l: y_pred.count(l) for l in labels}
    majority_frac = max(pred_dist.values()) / len(y_pred)
    collapsed = majority_frac > 0.97 or len(set(y_pred)) == 1
    return {
        "accuracy": accuracy,
        "perClass": {
            labels[i]: {"precision": precision[i], "recall": recall[i], "f1": f1[i], "support": int(support[i])}
            for i in range(len(labels))
        },
        "predDist": pred_dist,
        "collapsed": collapsed,
        "majorityFrac": majority_frac,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sev", choices=["sev"])
    parser.add_argument("--sample_size", type=int, default=300)
    args = parser.parse_args()

    test_csv = f"{args.task}_test.csv"
    results_path = RESULTS_DIR / f"{args.task}_classification_results.json"

    test_df = pd.read_csv(test_csv).sample(n=args.sample_size, random_state=42).reset_index(drop=True)
    with open(results_path) as f:
        all_results = json.load(f)

    n = len(test_df)
    baseline_results = all_results[:n]
    context_results = all_results[n : 2 * n]
    if len(baseline_results) != n or len(context_results) != n:
        raise SystemExit(
            f"Row-count mismatch: test_df has {n} rows but classification_results.json "
            f"has {len(all_results)} total (expected {2 * n}). Re-run 03_classify.py with "
            f"the same --sample_size before exporting."
        )

    retrieval_mod = load_retrieval_module()
    # Rebuild rather than unpickle artifacts/*_retriever.pkl: that file was
    # pickled while 02_retrieval.py ran as __main__, so unpickling it from
    # here fails ("Can't get attribute 'BugRetriever' on <module '__main__'
    # ...>") since this script is __main__ now. Rebuilding is deterministic
    # and only takes a few seconds for 12k rows.
    word_to_idx, embeddings = retrieval_mod.load_embeddings("vocab.lst", "embedding.npy")
    train_df = pd.read_csv(f"{args.task}_train.csv")
    retriever = retrieval_mod.BugRetriever(train_df, word_to_idx, embeddings)

    rows = []
    for i, row in test_df.iterrows():
        true_label = LABEL_NAMES[int(row["Label"])]
        b, c = baseline_results[i], context_results[i]
        if b["true_label"] != true_label or c["true_label"] != true_label:
            raise SystemExit(
                f"Alignment check failed at row {i}: test_df true label is "
                f"'{true_label}' but classification_results.json says baseline="
                f"'{b['true_label']}' context='{c['true_label']}'. Refusing to "
                f"export mismatched data — check that --sample_size and the "
                f"underlying CSV haven't changed since the classify run."
            )

        neighbors = retriever.retrieve(row["Description"], k=3)
        rows.append(
            {
                "id": int(i),
                "description": str(row["Description"]),
                "trueLabel": true_label,
                "baseline": {
                    "prediction": b["prediction"],
                    "correct": b["prediction"] == true_label,
                    "rawResponse": b.get("raw_response", ""),
                },
                "context": {
                    "prediction": c["prediction"],
                    "correct": c["prediction"] == true_label,
                    "rawResponse": c.get("raw_response", ""),
                    "neighbors": [
                        {
                            "description": str(nb["text"])[:400],
                            "label": LABEL_NAMES[nb["label"]],
                            "similarity": round(nb["similarity"], 4),
                        }
                        for nb in neighbors
                    ],
                },
                "agree": b["prediction"] == c["prediction"],
            }
        )

    summary = {
        "n": n,
        "trueDist": {l: sum(1 for r in rows if r["trueLabel"] == l) for l in LABEL_NAMES.values()},
        "baseline": compute_metrics(rows, "baseline"),
        "context": compute_metrics(rows, "context"),
    }

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=None, separators=(",", ":"))

    disagreements = sum(1 for r in rows if not r["agree"])
    context_wrong_baseline_right = sum(
        1 for r in rows if r["baseline"]["correct"] and not r["context"]["correct"]
    )
    print(f"Exported {n} rows to {out_path.resolve()}")
    print(f"  Disagreements (baseline != context): {disagreements}")
    print(f"  Context wrong, baseline right: {context_wrong_baseline_right}")
    print(f"  Baseline accuracy: {summary['baseline']['accuracy']:.1%}")
    print(f"  Context accuracy:  {summary['context']['accuracy']:.1%}")


if __name__ == "__main__":
    main()
