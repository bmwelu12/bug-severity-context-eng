"""
02_retrieval.py

Retrieval component: for a query bug description, find the k most similar
bugs in the training set and return them along with their known labels.

DEVIATION FROM ORIGINAL PLAN: rather than building TF-IDF from scratch, this
uses the pre-trained 100-dim word embeddings that shipped with the dataset
(embedding.npy + vocab.lst), trained on this exact corpus. A document vector
is the mean of its in-vocabulary word vectors. Similarity is cosine.

This is a better retrieval signal than a from-scratch TF-IDF would be, and
it makes use of an asset that was already sitting in the data. It also means
building the train-set doc-vector matrix once and reusing it, rather than
searching in raw text space.

Tested end-to-end against the real train/test data and real embeddings.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

TOKEN_RE = re.compile(r"[a-zA-Z]+")


def load_embeddings():
    vocab = [w.strip() for w in open(RAW_DIR / "vocab.lst", encoding="utf-8")]
    emb = np.load(RAW_DIR / "embedding.npy")
    assert len(vocab) == emb.shape[0], "vocab/embedding row mismatch"
    word2idx = {w: i for i, w in enumerate(vocab)}
    return word2idx, emb


def tokenize(text: str):
    return [t.lower() for t in TOKEN_RE.findall(text)]


def doc_vector(text: str, word2idx, emb):
    idxs = [word2idx[t] for t in tokenize(text) if t in word2idx]
    if not idxs:
        return np.zeros(emb.shape[1], dtype=np.float32)
    return emb[idxs].mean(axis=0)


def build_doc_matrix(descriptions, word2idx, emb):
    mat = np.vstack([doc_vector(d, word2idx, emb) for d in descriptions])
    # normalize for cosine similarity via dot product
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Retriever:
    def __init__(self, task: str = "sev"):
        self.word2idx, self.emb = load_embeddings()
        self.train = pd.read_csv(PROC_DIR / f"{task}_train_clean.csv")
        self.label_col = "Severity" if task == "sev" else "Fixing_time"
        self.train_matrix = build_doc_matrix(self.train["Description"].tolist(), self.word2idx, self.emb)

    def query(self, text: str, k: int = 3):
        qvec = doc_vector(text, self.word2idx, self.emb)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm
        sims = self.train_matrix @ qvec
        top_idx = np.argsort(-sims)[:k]
        results = []
        for i in top_idx:
            row = self.train.iloc[i]
            results.append({
                "similarity": float(sims[i]),
                "description": row["Description"][:500],
                "label_value": row[self.label_col],
                "label": int(row["Label"]),
            })
        return results


if __name__ == "__main__":
    # Real smoke test against real data: pull a handful of test examples
    # and confirm retrieval returns sensible, similar-looking neighbors.
    retriever = Retriever(task="sev")
    test = pd.read_csv(PROC_DIR / "sev_test_clean.csv").sample(3, random_state=1)
    for _, row in test.iterrows():
        print("=" * 80)
        print("QUERY:", row["Description"][:200])
        print("true label:", row["Label"], "(severity:", row["Severity"], ")")
        for r in retriever.query(row["Description"], k=3):
            print(f"  sim={r['similarity']:.3f} label={r['label']} ({r['label_value']}) :: {r['description'][:150]}")
