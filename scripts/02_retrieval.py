"""
02_retrieval.py

Uses the provided 100-dim word embeddings (embedding.npy + vocab.lst) to
build sentence-level representations via averaged word vectors, then does
cosine-similarity retrieval over the training set. This uses real
pre-trained embeddings rather than building TF-IDF from scratch, since
they were already provided and are a more meaningful semantic
representation than word-overlap counting.
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
import pickle

ARTIFACT_DIR = Path("./artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def load_embeddings(vocab_path: str, embedding_path: str):
    with open(vocab_path) as f:
        vocab = [line.strip() for line in f]
    embeddings = np.load(embedding_path)
    assert len(vocab) == embeddings.shape[0], (
        f"Vocab size {len(vocab)} does not match embedding rows {embeddings.shape[0]}"
    )
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return word_to_idx, embeddings


def strip_boilerplate(text: str) -> str:
    """
    Bugzilla exports prepend a near-identical metadata header to every
    report ("Created by X on <date>. Additional Details"), plus repeated
    "Updated by..." lines for each comment. Left in, this boilerplate
    dominates an averaged-embedding similarity score since it's shared
    across almost every report, regardless of what the bug actually is.
    Stripping it is necessary for retrieval to reflect bug content rather
    than export formatting.
    """
    text = re.sub(r"Created by .*? PDT", "", text)
    text = re.sub(r"Updated by .*? PDT", "", text)
    text = re.sub(r"Additional Details", "", text)
    text = re.sub(r"In reply to comment \d+", "", text)
    return text


def tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z']+", strip_boilerplate(text).lower())


def sentence_vector(text: str, word_to_idx: dict, embeddings: np.ndarray) -> np.ndarray:
    """
    Average word embedding over the tokens present in vocab. A reasonable
    first thing to try before reaching for a heavier sentence-transformer
    model, since it uses the embeddings already on hand.
    """
    tokens = tokenize(text)
    vecs = [embeddings[word_to_idx[t]] for t in tokens if t in word_to_idx]
    if not vecs:
        return np.zeros(embeddings.shape[1])
    return np.mean(vecs, axis=0)


class BugRetriever:
    def __init__(self, train_df: pd.DataFrame, word_to_idx: dict, embeddings: np.ndarray,
                 text_col: str = "Description", label_col: str = "Label"):
        self.train_df = train_df.reset_index(drop=True)
        self.word_to_idx = word_to_idx
        self.embeddings = embeddings
        self.text_col = text_col
        self.label_col = label_col

        print(f"Building sentence vectors for {len(self.train_df):,} training bugs...")
        self.doc_matrix = np.vstack([
            sentence_vector(str(t), word_to_idx, embeddings) for t in self.train_df[text_col]
        ])
        norms = np.linalg.norm(self.doc_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.doc_matrix_norm = self.doc_matrix / norms

    def retrieve(self, query_text: str, k: int = 3) -> list:
        q_vec = sentence_vector(query_text, self.word_to_idx, self.embeddings)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec_norm = q_vec / q_norm

        sims = self.doc_matrix_norm @ q_vec_norm
        top_k_idx = np.argsort(sims)[::-1][:k]

        results = []
        for idx in top_k_idx:
            results.append({
                "text": self.train_df.iloc[idx][self.text_col],
                "label": int(self.train_df.iloc[idx][self.label_col]),
                "similarity": float(sims[idx]),
            })
        return results

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def format_context_block(retrieved: list, label_names: dict) -> str:
    if not retrieved:
        return ""
    lines = ["Similar past bugs and their actual outcome:\n"]
    for i, r in enumerate(retrieved, 1):
        snippet = str(r["text"])[:250].replace("\n", " ")
        outcome = label_names.get(r["label"], str(r["label"]))
        lines.append('{}. "{}..." -> {} (similarity: {:.2f})'.format(i, snippet, outcome, r["similarity"]))
    return "\n".join(lines)


def main():
    word_to_idx, embeddings = load_embeddings(
        "./vocab.lst",
        "./embedding.npy",
    )
    print(f"Loaded embeddings: {embeddings.shape[0]:,} words, {embeddings.shape[1]} dims")

    for task in ["sev", "fix"]:
        train_df = pd.read_csv(f"./{task}_train.csv")
        retriever = BugRetriever(train_df, word_to_idx, embeddings)
        retriever.save(ARTIFACT_DIR / f"{task}_retriever.pkl")
        print(f"Saved {task} retriever to {ARTIFACT_DIR}/{task}_retriever.pkl")

        sample = train_df.iloc[0]["Description"]
        results = retriever.retrieve(sample, k=3)
        label_names = {0: "not severe", 1: "severe"} if task == "sev" else {0: "fast fix", 1: "slow fix"}
        print(format_context_block(results, label_names))
        print()


if __name__ == "__main__":
    main()
