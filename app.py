"""
Bug severity triage — baseline live demo + eval dashboard.

Deploy to a Hugging Face Space (Gradio SDK). Needs, somewhere findable
(checks data/raw/, then the repo root — HF's drag-and-drop folder upload
doesn't reliably preserve nested subfolders, so this tolerates a flattened
upload instead of requiring exact nesting):
  - sev_eval_summary.json   (eval dashboard tab; also checked under a
    results/ subfolder. Optional — the tab shows a message instead of
    crashing if it's missing)

Set ANTHROPIC_API_KEY as a Space secret (Settings -> Repository secrets).
Without it, the "Try it live" tab shows a note.

Baseline only — the context-engineered condition (retrieval over
sev_train.csv/embedding.npy/vocab.lst) was removed from this demo. It's
still in the actual experiment: see scripts/02_retrieval.py,
scripts/03_classify.py, and the real n=300 result in the GitHub repo's
README and results.html.
"""

import json
import os
import re
import time
from pathlib import Path

import gradio as gr
import pandas as pd

DATA_DIR = Path("data/raw")
RESULTS_DIR = DATA_DIR / "results"

# HF Spaces' drag-and-drop folder upload doesn't reliably preserve nested
# subfolders, so a fresh upload often lands this file flattened one or two
# levels up from where the repo puts it. Rather than make the user fight
# the upload UI into exact nesting, search the likely spots in order and
# use whichever actually has the file.
def find_file(filename, search_dirs):
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate
    return None


MODEL = "claude-haiku-4-5"
SYSTEM_PROMPT = (
    "You are triaging Bugzilla bug reports. Given a bug description, decide "
    "whether it is SEVERE (would be filed as blocker, critical, or major) or "
    "NOT SEVERE (minor, normal, or trivial). "
    "Respond with only a JSON object: {\"prediction\": \"severe\" or \"not severe\"}."
)
LABEL_NAMES = {0: "not severe", 1: "severe"}

anthropic_client = None
if os.environ.get("ANTHROPIC_API_KEY"):
    import anthropic

    anthropic_client = anthropic.Anthropic()


def build_prompt(description):
    return f"\nBug report:\n{description[:1500]}\n\nClassification:"


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


def format_result(raw, pred, elapsed_ms):
    if pred is None:
        badge = f"⚠️ **Could not parse response**\n\n```\n{raw}\n```"
    else:
        emoji = "🔴" if pred == "severe" else "🟢"
        badge = f"{emoji} **{pred.upper()}**"
    return f"{badge}\n\n_{elapsed_ms:.0f} ms_"


def classify(description):
    if not description.strip():
        return "_Enter a bug description above._"
    if anthropic_client is None:
        return "_Set `ANTHROPIC_API_KEY` as a Space secret to enable live classification._"

    # Catches anything from the API call itself (auth, workspace-scoping,
    # rate limits, network) so a failure shows up as readable text in the
    # output box instead of leaving it blank with the real error only in
    # the Space's container logs.
    try:
        raw, ms = call_claude(build_prompt(description))
    except Exception as e:
        return f"⚠️ **API call failed**\n\n```\n{type(e).__name__}: {e}\n```"
    return format_result(raw, parse_prediction(raw), ms)


EXAMPLES = [
    ["Using commercial build 2000 04 17 12 on mac I crash closing messenger. Launch messenger, select file close. Crash. MacsBug PowerPC illegal instruction."],
    ["The date format in the thread pane is displayed in localized format on a localized system, but in the message view pane the date is always in US format."],
    ["Open the prefs, select auto proxy, do a paste of a proxy you copied from an email message. Crash. I haven't tested the other platforms yet."],
]


def load_eval_summary():
    path = find_file("sev_eval_summary.json", [RESULTS_DIR, DATA_DIR, Path(".")])
    if path is None:
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
            "No sev_eval_summary.json found (checked data/raw/results/, "
            "data/raw/, and the repo root) — run scripts/03_classify.py and "
            "scripts/04_eval.py first, or copy the file from the GitHub "
            "repo into this Space.",
            None,
        )
    baseline_df = report_to_df(summary["baseline"])
    return "", baseline_df


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
            "# 🐛 Bug Severity Triage\n"
            "Claude Haiku 4.5 classifying real Bugzilla bug reports as "
            "severe (blocker/critical/major) or not severe "
            "(minor/normal/trivial)."
        )

    with gr.Tab("🔴 Try it live"):
        gr.Markdown(
            "Paste a bug report below, or pick an example, and see how "
            "Claude Haiku 4.5 classifies it.",
            elem_classes="section-header",
        )
        desc_in = gr.Textbox(
            label="Bug description", lines=5,
            placeholder="Paste a bug report here...",
        )
        run_btn = gr.Button("🚀 Classify", variant="primary", size="lg")
        gr.Examples(examples=EXAMPLES, inputs=[desc_in], label="Example bug reports")

        gr.Markdown("### Result", elem_classes="section-header")
        with gr.Group(elem_classes="model-card"):
            base_out = gr.Markdown()

        run_btn.click(classify, inputs=[desc_in], outputs=[base_out])

    with gr.Tab("📊 Eval dashboard"):
        missing_msg, baseline_df = build_dashboard()
        if missing_msg:
            gr.Markdown(f"⚠️ {missing_msg}")
        else:
            gr.Markdown(
                "### Real n=300 result (Claude Haiku 4.5, baseline)\n"
                "75.3% accuracy against a 77.2% majority-class baseline — "
                "did not collapse to the majority class.",
                elem_classes="section-header",
            )
            gr.Dataframe(baseline_df, interactive=False)

if __name__ == "__main__":
    demo.launch(show_error=True)
