"""
03_classify.py

Run a sample of the held-out test set through two conditions, for a chosen
task (severity or fix-time):

  1. baseline:            bug description alone
  2. context_engineered:  bug description + 3 retrieved similar bugs and
                           their actual outcomes

The critical number to watch isn't accuracy alone. It's whether either
condition just predicts the majority class every time:
  - severity majority class (0, not severe): 77.2% of test set
  - fix-time majority class (0, fast fix):    79.4% of test set

Any accuracy near those numbers is a warning sign, not a win, exactly the
lesson from the ticket triage project's collapse.
"""

import pandas as pd
import json
import re
import time
import argparse
from pathlib import Path
from anthropic import Anthropic
from tqdm import tqdm

from importlib import import_module
retrieval_module = import_module("02_retrieval")
BugRetriever = retrieval_module.BugRetriever
format_context_block = retrieval_module.format_context_block

ARTIFACT_DIR = Path("./artifacts")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

client = Anthropic()
MODEL = "claude-haiku-4-5"

TASK_CONFIG = {
    "sev": {
        "test_csv": "./sev_test.csv",
        "retriever_path": ARTIFACT_DIR / "sev_retriever.pkl",
        "label_names": {0: "not severe", 1: "severe"},
        "question": "Is this bug report describing a SEVERE issue (major/critical) or not?",
    },
    "fix": {
        "test_csv": "./fix_test.csv",
        "retriever_path": ARTIFACT_DIR / "fix_retriever.pkl",
        "label_names": {0: "fast fix", 1: "slow fix"},
        "question": "Will this bug take a LONG TIME to fix (slow fix) or get resolved quickly (fast fix)?",
    },
}


def build_prompt(bug_text: str, question: str, label_names: dict, context_block: str = "") -> str:
    context_section = f"\n{context_block}\n" if context_block else ""
    options = " or ".join(f'"{v}"' for v in label_names.values())
    return f"""{question}
{context_section}
Bug report:
\"\"\"{bug_text}\"\"\"

Respond with only a JSON object: {{"prediction": {options}}}"""


def extract_json_object(text: str) -> str:
    """Claude Haiku often wraps JSON replies in a ```json ... ``` fence
    (or adds trailing prose after truncation). Pull out the first {...}
    block rather than assuming the raw text is bare JSON."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    return match.group(0) if match else text


def classify_one(bug_text: str, question: str, label_names: dict, context_block: str = "") -> dict:
    prompt = build_prompt(bug_text, question, label_names, context_block)
    start = time.time()
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.time() - start) * 1000

    raw_text = response.content[0].text.strip()
    valid_labels = list(label_names.values())
    try:
        parsed = json.loads(extract_json_object(raw_text))
        prediction_text = parsed.get("prediction", "").lower()
        prediction = next((v for v in valid_labels if v.lower() == prediction_text), None)
        parse_failed = prediction is None
    except (json.JSONDecodeError, AttributeError):
        prediction = None
        parse_failed = True

    return {
        "prediction": prediction,
        "parse_failed": parse_failed,
        "latency_ms": latency_ms,
        "raw_response": raw_text,
    }


def run_condition(test_df: pd.DataFrame, retriever: BugRetriever, config: dict, use_context: bool) -> list:
    results = []
    label = "context_engineered" if use_context else "baseline"

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{label}"):
        context_block = ""
        if use_context:
            retrieved = retriever.retrieve(row["Description"], k=3)
            context_block = format_context_block(retrieved, config["label_names"])

        outcome = classify_one(row["Description"], config["question"], config["label_names"], context_block)
        outcome["true_label"] = config["label_names"][int(row["Label"])]
        outcome["condition"] = label
        results.append(outcome)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sev", "fix"], default="sev")
    parser.add_argument("--sample_size", type=int, default=200,
                         help="Number of test rows per condition. Full test sets are 4,000+ rows; "
                              "a sample keeps API cost and runtime reasonable for a first pass.")
    args = parser.parse_args()

    config = TASK_CONFIG[args.task]
    test_df = pd.read_csv(config["test_csv"]).sample(n=args.sample_size, random_state=42)
    retriever = BugRetriever.load(config["retriever_path"])

    print(f"Task: {args.task} | Sample size: {len(test_df)} | Majority class check applies")

    baseline_results = run_condition(test_df, retriever, config, use_context=False)
    context_results = run_condition(test_df, retriever, config, use_context=True)

    all_results = baseline_results + context_results
    out_path = RESULTS_DIR / f"{args.task}_classification_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved {len(all_results)} results to {out_path}")


if __name__ == "__main__":
    main()
