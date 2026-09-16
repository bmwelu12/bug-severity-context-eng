# Bug severity triage — context engineering experiment

Same question as the [ticket-triage LoRA project](https://github.com/bmwelu12/ticket-triage-lora-finetuning),
approached a different way: instead of fine-tuning a model, this tests
whether giving Claude **retrieved similar past examples** at inference time
(context engineering / lightweight RAG) beats giving it nothing, using the
same eval discipline — per-class precision/recall/F1, not just accuracy,
and an explicit check for collapse to the majority class.

## Task

Binary classification on real Bugzilla bug reports: is a bug **severe**
(blocker/critical/major) or **not severe** (minor/normal/trivial)? Class
split is roughly 76/24, so overall accuracy alone is a bad signal — a model
that always predicts "not severe" already scores ~76%.

- `--mode baseline`: the model sees only the raw bug description.
- `--mode context`: the model also sees the top-k most similar past bugs
  (retrieved via pre-trained word embeddings) and how they were actually
  labeled.

## What's tested vs. not (as of the initial build)

- `01_data_prep.py` — run against the real CSVs, output verified (27,998
  train / 4,427 test rows, label balance printed and matches source data).
- `02_retrieval.py` — run end-to-end against real embeddings and real test
  descriptions. Retrieved neighbors are topically similar (same feature
  area) but similarity scores are uniformly high (0.97-0.99) and don't
  track severity well — one query labeled "blocker" pulled back a "normal"
  neighbor. Worth watching for in the eval: mean-pooled embeddings capture
  topic, not severity, so the context-engineered run might not help much.
- `03_classify.py` — prompt construction, retrieval hookup, and CSV output
  verified with `--dry-run` (mocked responses, no API calls). Actual model
  behavior on real data needs a real `ANTHROPIC_API_KEY` and real spend —
  **not yet run**.
- `04_eval.py` — verified against real test labels with synthetic
  prediction sets (a forced majority-class collapse, correctly flagged;
  a noisy-but-mixed set, correctly not flagged).

This mirrors the LoRA project's workflow: the heavy/paid execution
(real API calls against the full test set) happens outside a chat session,
in the notebook, with your own key.

## Data

Not committed to this repo (same as the LoRA project — data lives locally,
not in git). You need these four files in `data/raw/`:

- `sev_train.csv`, `sev_test.csv` — pre-split bug reports with `Description`,
  `Severity`, `Label` columns.
- `embedding.npy`, `vocab.lst` — a pre-trained 100-dim word embedding table
  fit on this corpus, used by the retriever (mean-pooled doc vectors,
  cosine similarity) instead of building TF-IDF from scratch.

## How to run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here

python3 scripts/01_data_prep.py --task sev

# cheap first pass before spending on the full 4,427-row test set
python3 scripts/03_classify.py --mode baseline --n 300
python3 scripts/03_classify.py --mode context --n 300 --k 3

python3 scripts/04_eval.py --task sev
```

Drop `--n 300` once the small run looks sane, to get the full test set.
Or run `bug_severity_context_eng.ipynb` in Colab, which walks through the
same steps with prompts for uploading `data/raw/` and entering your API key.

## The honest question this is trying to answer

Retrieved neighbors are topically similar but not obviously
severity-similar. Does giving Claude those neighbors actually help it call
severity, or does it just add noise? `04_eval.py`'s collapse check plus
per-class F1 on the *severe* class (not overall accuracy) is what answers
that. Either outcome — context helps, context is neutral, or context hurts
— is a legitimate, reportable finding.
