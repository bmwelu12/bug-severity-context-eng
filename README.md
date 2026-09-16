# Bug severity triage — context engineering experiment

Same question as the [ticket-triage LoRA project](https://github.com/bmwelu12/ticket-triage-lora-finetuning),
approached a different way: instead of fine-tuning a model, this tests
whether giving Claude **retrieved similar past examples** at inference time
(context engineering / lightweight RAG) beats giving it nothing.

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

## Real result (n=300, `--task sev`)

|  | Accuracy | vs. 77.2% majority baseline | Recall: not severe | Recall: severe |
|---|---|---|---|---|
| Baseline | 75.3% | −1.9 pts | 0.740 | **0.792** |
| Context-engineered | 80.3% | +3.1 pts | 0.830 | **0.727** |

Neither run collapsed to the majority class. But the result is mixed, not a
clean win: context engineering raised overall accuracy by pushing the model
to predict "not severe" more often (recall on that class rose 0.740 ->
0.830) — which, because "not severe" is the majority class, mechanically
improves accuracy. The cost is recall on the class that actually matters
for triage: **severe-bug recall dropped from 0.792 to 0.727** under
context. The retrieved neighbors are topically similar (cosine similarity
0.97-1.00 regardless of label) but not reliably severity-similar, so the
context sometimes nudges the model toward the topic cluster's typical
severity rather than the true one — here, toward under-calling severity.

This was a 300-row first pass, not the full test set — treat the direction
as suggestive, not final. Full `04_eval.py` output and raw predictions
(including each response's raw text) are in `data/raw/results/`.

One bug worth flagging for anyone extending this: `claude-haiku-4-5` wraps
JSON replies in a ` ```json ` fence, so a plain `json.loads()` on the raw
response fails 100% of the time. `classify_one()` in `03_classify.py`
extracts the `{...}` object first — the first real run silently produced
zero valid predictions on both conditions until this was fixed.
