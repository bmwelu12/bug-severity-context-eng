# Bug severity triage — context engineering experiment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmwelu12/bug-severity-context-eng/blob/main/bug_severity_context_eng.ipynb)

## Task

Binary classification on real Bugzilla bug reports: is a bug **severe**
(blocker/critical/major) or **not severe** (minor/normal/trivial)? Class
split is 77.2% / 22.8% in the test set, so overall accuracy alone is a bad
signal — a model that always predicts "not severe" already scores 77.2%.

`03_classify.py` runs both conditions on the same sample in one invocation:

- **baseline**: the model sees only the raw bug description.
- **context_engineered**: the model also sees the top-3 most similar past
  bugs (retrieved via pre-trained word embeddings) and how they were
  actually labeled.

A `fix_{train,test}.csv` pair also ships (is a bug slow to fix, >~100 days?)
but `fix_train.csv` has no `Label` column, only a raw `Fixing_time` — the
`fix` task isn't runnable until that's derived and isn't part of the result
below.

## What's genuinely tested vs. not

- `01_data_prep.py` — an earlier draft of this pipeline cleaned raw CSVs
  into `data/processed/*_clean.csv` before classification. The current
  `02_retrieval.py`/`03_classify.py` read `data/raw/*.csv` directly, so
  this script is **orphaned** — still in `scripts/`, not called by anything.
- `02_retrieval.py` — run end-to-end against the real embeddings and real
  training data (12,000 rows). Confirmed working: builds the `sev`
  retriever and pickles it. Loops over `sev` then `fix`; without
  `fix_train.csv`/`fix_test.csv` having a proper `Label` column it errors
  on the `fix` half — expected, doesn't block `sev`.
- `03_classify.py` — run for real against `claude-haiku-4-5`, real spend,
  600 calls. The **first real run silently produced 0/600 valid
  predictions**: Haiku wraps JSON replies in a ` ```json ` fence, and the
  original `classify_one()` called `json.loads()` on the raw text
  directly. Fixed to extract the `{...}` object first; the second run
  parsed cleanly (0 failures) and produced the result above.
- `04_eval.py` — run against the real corrected predictions; also
  previously verified with synthetic majority-collapse and mixed
  prediction sets to confirm the collapse check fires/doesn't fire
  correctly.
- `fix` task — not runnable as shipped: `fix_train.csv` has a raw
  `Fixing_time` column but no `Label`, while `fix_test.csv` has both.
  Deriving train's `Label` from `Fixing_time` would need the same
  threshold that produced test's `Label`, which hasn't been verified.

## Data

`data/raw/` holds `sev_train.csv`, `sev_test.csv`, `fix_train.csv`,
`fix_test.csv` (all with a `Description` column, ~9-25MB each), plus
`embedding.npy` (100-dim word embeddings pre-trained on this corpus,
~18.5MB) and `vocab.lst`. All committed as plain files — none of them are
close to GitHub's 100MB limit, so Git LFS isn't needed here.

## How to run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # needs workspace-scoped access

cd data/raw
python3 ../../scripts/02_retrieval.py            # builds + pickles the retriever(s)
python3 ../../scripts/03_classify.py --task sev --sample_size 300
python3 ../../scripts/04_eval.py --task sev
```

Scripts read data files relative to the current directory (`./sev_test.csv`
etc.) and write `artifacts/` (pickled retrievers) and `results/`
(predictions + eval summary) the same way — run them from `data/raw/`.
Drop `--sample_size` to run the full 4,427-row test set (2x the API cost of
the n=300 pass). Or run `bug_severity_context_eng.ipynb` in Colab.

## Live demo

`app.py` is a Gradio app (matching the [ticket-triage project](https://github.com/bmwelu12/ticket-triage-lora-finetuning)'s
HF Space pattern): a "Try it live" tab where you paste a bug report and see
baseline vs. context-engineered classification side by side with the actual
retrieved neighbors, plus an "Eval dashboard" tab rendering the real
`sev_eval_summary.json`. Tested locally end-to-end (real retriever, real
Claude calls) — screenshots and a walkthrough are in the PR/commit history.

To run locally:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python3 app.py
```

To deploy to a Hugging Face Space: create a Gradio Space, push this repo's
contents to it (`app.py`, `requirements.txt`, and `data/raw/` — the retriever
needs `sev_train.csv`, `embedding.npy`, `vocab.lst`; the dashboard tab needs
`data/raw/results/sev_eval_summary.json`), and set `ANTHROPIC_API_KEY` under
Settings → Repository secrets. Without that secret the "Try it live" tab
still loads but shows a note instead of calling the API.

## Real result (n=300, `--task sev`)

**[Open the full results page →](results.html)** (predicted-label
distributions, confusion matrices, per-class precision/recall/F1, and the
collapse-check verdict for both conditions, rendered from the actual run).

| | not severe P / R / F1 | severe P / R / F1 | Accuracy | Macro F1 |
|---|---|---|---|---|
| Baseline | 91.2 / 74.0 / 81.7 | 51.3 / **79.2** / 62.2 | 75.3% | 72.0% |
| Context-engineered | 89.8 / 83.0 / 86.3 | 59.6 / **72.7** / **65.5** | 80.3% | 75.9% |

Neither run collapsed to the majority class (77.2% "not severe"). Reading
past accuracy alone: context-engineering raised F1 on **both** classes
(severe F1 62.2 → 65.5, not-severe F1 81.7 → 86.3) and raised overall
accuracy 75.3% → 80.3% — a real, if modest, net positive on this sample.
But it got there partly by calling "severe" less often: **severe-class
recall dropped from 79.2% to 72.7%**, traded for a large precision gain
(51.3% → 59.6%). For a bug-triage system, missing more actually-severe bugs
to reduce false alarms is a real cost that the aggregate F1/accuracy
numbers understate. The retrieved neighbors are topically similar (cosine
similarity 0.97-1.00 regardless of label) but not reliably
severity-similar, so context sometimes nudges the model toward the topic
cluster's typical severity rather than the true one — here, toward
under-calling severity on the margin.

This was a 300-row first pass, not the full test set — treat the direction
as suggestive, not final. Full `04_eval.py` output and raw predictions
(including each response's raw text) are in `data/raw/results/`.

One bug worth flagging for anyone extending this: `claude-haiku-4-5` wraps
JSON replies in a ` ```json ` fence, so a plain `json.loads()` on the raw
response fails 100% of the time. `classify_one()` in `03_classify.py`
extracts the `{...}` object first — the first real run silently produced
zero valid predictions on both conditions until this was fixed.
