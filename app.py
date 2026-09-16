"""
Bug severity triage — live demo + eval dashboard.

Deploy to a Hugging Face Space (Gradio SDK). Expects, alongside this file,
the same layout as the GitHub repo:
  - data/raw/sev_train.csv, embedding.npy, vocab.lst   (builds the retriever)
  - data/raw/results/sev_eval_summary.json             (eval dashboard tab;
    optional — the tab shows a message instead of crashing if it's missing)

Set ANTHROPIC_API_KEY as a Space secret (Settings -> Repository secrets).
Without it, the "Try it live" tab shows a note and only the dashboard works.
"""

import json
import os
import re
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")
RESULTS_DIR = DATA_DIR / "results"

MODEL = "claude-haiku-4-5"
SYSTEM_PROMPT = (
    "You are triaging Bugzilla bug reports. Given a bug description, decide "
    "whether it is SEVERE (would be filed as blocker, critical, or major) or "
    "NOT SEVERE (minor, normal, or trivial). "
    "Respond with only a JSON object: {\"prediction\": \"severe\" or \"not severe\"}."
)
LABEL_NAMES = {0: "not severe", 1: "severe"}

TOKEN_RE = re.compile(r"[a-zA-Z']+")


def strip_boilerplate(text: str) -> str:
    text = re.sub(r"Created by .*? PDT", "", text)
    text = re.sub(r"Updated by .*? PDT", "", text)
    text = re.sub(r"Additional Details", "", text)
    text = re.sub(r"In reply to comment \d+", "", text)
    return text


def tokenize(text: str) -> list:
    return TOKEN_RE.findall(strip_boilerplate(text).lower())


# --- retriever: mirrors scripts/02_retrieval.py, inlined so this file is a
# single self-contained Space app ---
class BugRetriever:
    def __init__(self, train_df, word_to_idx, embeddings):
        self.train_df = train_df.reset_index(drop=True)
        self.word_to_idx = word_to_idx
        self.embeddings = embeddings
        self.doc_matrix = np.vstack(
            [self._vec(str(t)) for t in self.train_df["Description"]]
        )
        norms = np.linalg.norm(self.doc_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.doc_matrix_norm = self.doc_matrix / norms

    def _vec(self, text):
        vecs = [self.embeddings[self.word_to_idx[t]] for t in tokenize(text) if t in self.word_to_idx]
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.embeddings.shape[1])

    def retrieve(self, query_text, k=3):
        q = self._vec(query_text)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = q / norm
        sims = self.doc_matrix_norm @ q
        top_idx = np.argsort(sims)[::-1][:k]
        return [
            {
                "text": self.train_df.iloc[i]["Description"],
                "label": int(self.train_df.iloc[i]["Label"]),
                "similarity": float(sims[i]),
            }
            for i in top_idx
        ]


def format_context_block(retrieved):
    lines = ["Similar past bugs and their actual outcome:\n"]
    for i, r in enumerate(retrieved, 1):
        snippet = str(r["text"])[:250].replace("\n", " ")
        outcome = LABEL_NAMES[r["label"]]
        lines.append(f'{i}. "{snippet}..." -> {outcome} (similarity: {r["similarity"]:.2f})')
    return "\n".join(lines)


retriever = None
if (DATA_DIR / "vocab.lst").exists() and (DATA_DIR / "embedding.npy").exists() and (DATA_DIR / "sev_train.csv").exists():
    print("Building severity retriever...")
    vocab = [w.strip() for w in open(DATA_DIR / "vocab.lst", encoding="utf-8")]
    embeddings = np.load(DATA_DIR / "embedding.npy")
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    train_df = pd.read_csv(DATA_DIR / "sev_train.csv")
    retriever = BugRetriever(train_df, word_to_idx, embeddings)
    print(f"Retriever ready: {len(train_df):,} training bugs, {embeddings.shape[1]}-dim vectors.")
else:
    print("Retriever data not found under data/raw/ — 'Try it live' tab will be disabled.")

anthropic_client = None
if os.environ.get("ANTHROPIC_API_KEY"):
    import anthropic

    anthropic_client = anthropic.Anthropic()


def build_prompt(description, context_block=None):
    context_section = f"\n{context_block}\n" if context_block else ""
    return f"{context_section}\nBug report:\n{description[:1500]}\n\nClassification:"


def call_claude(prompt):
    start = time.perf_counter()
    resp = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=50,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return resp.content[0].text, elapsed_ms


def extract_json_object(text):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    return match.group(0) if match else text


def parse_prediction(raw_text):
    try:
        parsed = json.loads(extract_json_object(raw_text))
        pred = str(parsed.get("prediction", "")).lower().strip()
        if pred in ("severe", "not severe"):
            return pred
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def format_result(raw, pred, elapsed_ms, neighbors=None):
    if pred is None:
        badge = f"⚠️ **Could not parse response**\n\n```\n{raw}\n```"
    else:
        emoji = "🔴" if pred == "severe" else "🟢"
        badge = f"{emoji} **{pred.upper()}**"
    lines = [badge, f"_{elapsed_ms:.0f} ms_"]
    if neighbors:
        lines.append("\n**Retrieved neighbors:**")
        for n in neighbors:
            snippet = str(n["text"])[:140].replace("\n", " ")
            lines.append(f"- sim={n['similarity']:.2f}, {LABEL_NAMES[n['label']]}: {snippet}...")
    return "\n\n".join(lines)


def classify(description):
    if not description.strip():
        empty = "_Enter a bug description above._"
        return empty, empty
    if anthropic_client is None:
        note = "_Set `ANTHROPIC_API_KEY` as a Space secret to enable live classification._"
        return note, note

    base_raw, base_ms = call_claude(build_prompt(description))
    base_out = format_result(base_raw, parse_prediction(base_raw), base_ms)

    if retriever is not None:
        neighbors = retriever.retrieve(description, k=3)
        context_block = format_context_block(neighbors)
        ctx_raw, ctx_ms = call_claude(build_prompt(description, context_block))
        ctx_out = format_result(ctx_raw, parse_prediction(ctx_raw), ctx_ms, neighbors)
    else:
        ctx_out = "_Retriever data not found — see app.py header for what to add to this Space._"

    return base_out, ctx_out


EXAMPLES = [
    ["Using commercial build 2000 04 17 12 on mac I crash closing messenger. Launch messenger, select file close. Crash. MacsBug PowerPC illegal instruction."],
    ["The date format in the thread pane is displayed in localized format on a localized system, but in the message view pane the date is always in US format."],
    ["Open the prefs, select auto proxy, do a paste of a proxy you copied from an email message. Crash. I haven't tested the other platforms yet."],
]


def load_eval_summary():
    path = RESULTS_DIR / "sev_eval_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def report_to_df(report):
    rows = []
    for label, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "class": label,
                "precision": round(metrics.get("precision", 0), 3),
                "recall": round(metrics.get("recall", 0), 3),
                "f1-score": round(metrics.get("f1-score", 0), 3),
                "support": int(metrics.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


def build_dashboard():
    summary = load_eval_summary()
    if summary is None:
        return (
            "No sev_eval_summary.json found under data/raw/results/ — run "
            "scripts/03_classify.py and scripts/04_eval.py first, or copy "
            "the file from the GitHub repo into this Space.",
            None,
            None,
        )
    baseline_df = report_to_df(summary["baseline"])
    context_df = report_to_df(summary["context_engineered"])
    return "", baseline_df, context_df


CUSTOM_CSS = """
.gradio-container {max-width: 1000px !important; margin: auto;}
.hero {text-align: center; padding: 8px 0 4px 0;}
.hero h1 {margin-bottom: 4px;}
.model-card {border-radius: 12px; padding: 16px; min-height: 160px;}
.section-header {margin-top: 8px; margin-bottom: 4px;}
footer {display: none !important;}
"""

with gr.Blocks(
    title="Bug Severity Triage",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"),
    css=CUSTOM_CSS,
) as demo:
    with gr.Column(elem_classes="hero"):
        gr.Markdown(
            "# 🐛 Bug Severity Triage — Context Engineering\n"
            "Does giving Claude retrieved similar past bugs improve severity "
            "classification versus a no-context baseline? Real Bugzilla bug "
            "reports, Claude Haiku 4.5."
        )

    with gr.Tab("🔴 Try it live"):
        gr.Markdown(
            "Paste a bug report below, or pick an example, and see how the "
            "baseline and context-engineered conditions classify it side by side.",
            elem_classes="section-header",
        )
        desc_in = gr.Textbox(
            label="Bug description", lines=5,
            placeholder="Paste a bug report here...",
        )
        run_btn = gr.Button("🚀 Classify", variant="primary", size="lg")
        gr.Examples(examples=EXAMPLES, inputs=[desc_in], label="Example bug reports")

        gr.Markdown("### Results", elem_classes="section-header")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 🔹 Baseline (no context)")
                with gr.Group(elem_classes="model-card"):
                    base_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("#### 🟠 Context-engineered (+3 similar bugs)")
                with gr.Group(elem_classes="model-card"):
                    ctx_out = gr.Markdown()

        run_btn.click(classify, inputs=[desc_in], outputs=[base_out, ctx_out])

    with gr.Tab("📊 Eval dashboard"):
        missing_msg, baseline_df, context_df = build_dashboard()
        if missing_msg:
            gr.Markdown(f"⚠️ {missing_msg}")
        else:
            gr.Markdown(
                "### Real n=300 result\n"
                "Context-engineering raised accuracy (75.3% → 80.3%) and F1 on "
                "both classes, but severe-class **recall** dropped 79.2% → "
                "72.7% — it called \"severe\" less often. Neither run "
                "collapsed to the majority class (77.2% not severe).",
                elem_classes="section-header",
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Baseline**")
                    gr.Dataframe(baseline_df, interactive=False)
                with gr.Column():
                    gr.Markdown("**Context-engineered**")
                    gr.Dataframe(context_df, interactive=False)

if __name__ == "__main__":
    demo.launch()
