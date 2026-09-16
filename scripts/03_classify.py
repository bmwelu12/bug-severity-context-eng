"""
03_classify.py

Runs two versions of the severity classifier over the test set using the
Anthropic API (Claude Haiku, matching the benchmark model from the ticket
triage project):

  --mode baseline : sees only the raw bug description
  --mode context  : sees the description PLUS the top-k retrieved similar
                    bugs (from 02_retrieval.py) and their known severity

NOT independently tested here: this needs ANTHROPIC_API_KEY and real API
spend, which isn't available in this environment. The prompt formatting and
response parsing were dry-run tested with mocked responses (see
`--dry-run`), but the actual model behavior is unverified until you run it.

Cost note: full test set is 4427 rows x 2 modes. Use --n to subsample
(e.g. --n 300) for a first pass before running the whole thing.
"""
import argparse
import json
import os
import time
import pandas as pd
from pathlib import Path

PROC_DIR = Path("data/processed")

SYSTEM_PROMPT = (
    "You are triaging Bugzilla bug reports. Given a bug description, decide "
    "whether it is SEVERE (would be filed as blocker, critical, or major) or "
    "NOT SEVERE (minor, normal, or trivial). "
    "Respond with only one word: SEVERE or NOT_SEVERE."
)


def build_baseline_prompt(description: str) -> str:
    return f"Bug report:\n{description[:1500]}\n\nClassification:"


def build_context_prompt(description: str, neighbors: list) -> str:
    context_lines = []
    for n in neighbors:
        label_str = "SEVERE" if n["label"] == 1 else "NOT_SEVERE"
        context_lines.append(
            f"- Similar past bug (was labeled {label_str}, severity={n['label_value']}): "
            f"{n['description'][:300]}"
        )
    context_block = "\n".join(context_lines)
    return (
        f"Here are similar past bug reports and how they were actually triaged:\n"
        f"{context_block}\n\n"
        f"New bug report:\n{description[:1500]}\n\nClassification:"
    )


def parse_response(text: str) -> int:
    text = text.strip().upper()
    if "NOT_SEVERE" in text or "NOT SEVERE" in text:
        return 0
    if "SEVERE" in text:
        return 1
    return -1  # unparseable, surfaced in eval as its own bucket


def call_claude(client, system: str, user_prompt: str, model: str = "claude-haiku-4-5-20251001") -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=10,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "context"], required=True)
    parser.add_argument("--task", choices=["sev"], default="sev")
    parser.add_argument("--n", type=int, default=None, help="subsample test set to N rows")
    parser.add_argument("--k", type=int, default=3, help="neighbors to retrieve for context mode")
    parser.add_argument("--dry-run", action="store_true",
                         help="build prompts and parse mocked responses, no API calls")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    test = pd.read_csv(PROC_DIR / f"{args.task}_test_clean.csv")
    if args.n:
        test = test.sample(n=min(args.n, len(test)), random_state=42).reset_index(drop=True)

    retriever = None
    if args.mode == "context":
        # module name starts with a digit, so load it by path rather than import
        import importlib.util
        spec = importlib.util.spec_from_file_location("retrieval_mod", "scripts/02_retrieval.py")
        retrieval_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(retrieval_mod)
        retriever = retrieval_mod.Retriever(task=args.task)

    client = None
    if not args.dry_run:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    results = []
    for i, row in test.iterrows():
        desc = row["Description"]
        if args.mode == "baseline":
            prompt = build_baseline_prompt(desc)
        else:
            neighbors = retriever.query(desc, k=args.k)
            prompt = build_context_prompt(desc, neighbors)

        if args.dry_run:
            # mocked response for pipeline testing without API spend
            raw_response = "NOT_SEVERE"
        else:
            raw_response = call_claude(client, SYSTEM_PROMPT, prompt)
            time.sleep(0.05)  # light rate-limit courtesy

        pred = parse_response(raw_response)
        results.append({
            "true_label": int(row["Label"]),
            "pred_label": pred,
            "raw_response": raw_response,
        })

        if i % 50 == 0:
            print(f"[{args.mode}] {i}/{len(test)}")

    out_path = args.out or f"data/processed/{args.task}_{args.mode}_predictions.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
