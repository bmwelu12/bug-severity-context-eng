"""
Bug severity triage — live demo + eval dashboard.

Deploy to a Hugging Face Space (Gradio SDK). "Try it live" runs base
Qwen2.5-0.5B-Instruct (zero-shot, no fine-tuning) side by side with
Claude Haiku 4.5 — same comparison shape as the ticket-triage-lora-
finetuning Space this app shares hardware with (ZeroGPU), just without
the LoRA adapter column since this project never fine-tunes a model.

Needs, somewhere findable (checks data/raw/, then the repo root — HF's
drag-and-drop folder upload doesn't reliably preserve nested subfolders,
so this tolerates a flattened upload instead of requiring exact nesting):
  - sev_eval_summary.json   (eval dashboard tab; also checked under a
    results/ subfolder. Optional — the tab shows a message instead of
    crashing if it's missing)

Set ANTHROPIC_API_KEY as a Space secret (Settings -> Repository secrets).
Without it, the Claude column shows a note; the Qwen column still works.
"""

import json
import os
import re
import time
from pathlib import Path

import gradio as gr
import pandas as pd
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

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


CLAUDE_MODEL = "claude-haiku-4-5"
SYSTEM_PROMPT = (
    "You are triaging Bugzilla bug reports. Given a bug description, decide "
    "whether it is SEVERE (would be filed as blocker, critical, or major) or "
    "NOT SEVERE (minor, normal, or trivial). "
    "Respond with only a JSON object: {\"prediction\": \"severe\" or \"not severe\"}."
)
LABEL_NAMES = {0: "not severe", 1: "severe"}


# Register the ZeroGPU-decorated function BEFORE any model loading below.
# ZeroGPU detects @spaces.GPU by looking at the function directly wired to
# a Gradio event (classify, via run_btn.click below) -- not just any
# decorated function that happens to exist in the module. It also detects
# by importing this file, so if model loading throws for any reason,
# Python never reaches a decorator defined after it, and HF shows the
# generic "No @spaces.GPU function detected" message instead of the real
# traceback. Defining generate_qwen here (undecorated) keeps that loading
# order; the decorator itself goes on classify() further down.
def generate_qwen(prompt, max_new_tokens=20):
    # This function body runs with a real GPU attached, for the duration of
    # this call only (that's the whole point of ZeroGPU). Move the model to
    # cuda here rather than at load time.
    local_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = qwen_model.to(local_device)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = qwen_tokenizer([text], return_tensors="pt").to(local_device)
    start_t = time.perf_counter()
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed_ms = (time.perf_counter() - start_t) * 1000
    out_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    raw = qwen_tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    return raw, elapsed_ms


# ZeroGPU only attaches a GPU inside functions decorated with @spaces.GPU —
# at import time (this code, right here) there is no CUDA device available,
# even though one will exist once a decorated function is actually called.
# So: load everything on CPU now, and move to "cuda" inside the decorated
# call in generate_qwen() above.
print("Loading base Qwen model...")
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, torch_dtype="auto")
print("Qwen ready.")

anthropic_client = None
if os.environ.get("ANTHROPIC_API_KEY"):
    import anthropic

    anthropic_client = anthropic.Anthropic()


def build_prompt(description):
    return f"\nBug report:\n{description[:1500]}\n\nClassification:"


def call_claude(prompt):
    start = time.perf_counter()
    resp = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
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


@spaces.GPU
def classify(description):
    if not description.strip():
        empty = "_Enter a bug description above._"
        return empty, empty

    prompt = build_prompt(description)

    try:
        qwen_raw, qwen_ms = generate_qwen(prompt)
        qwen_out = format_result(qwen_raw, parse_prediction(qwen_raw), qwen_ms)
    except Exception as e:
        qwen_out = f"⚠️ **Generation failed**\n\n```\n{type(e).__name__}: {e}\n```"

    if anthropic_client is None:
        claude_out = "_Set `ANTHROPIC_API_KEY` as a Space secret to enable this column._"
    else:
        # Catches anything from the API call itself (auth, workspace-scoping,
        # rate limits, network) so a failure shows up as readable text in the
        # output box instead of leaving it blank with the real error only in
        # the Space's container logs.
        try:
            claude_raw, claude_ms = call_claude(prompt)
            claude_out = format_result(claude_raw, parse_prediction(claude_raw), claude_ms)
        except Exception as e:
            claude_out = f"⚠️ **API call failed**\n\n```\n{type(e).__name__}: {e}\n```"

    return qwen_out, claude_out


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
            "Base Qwen2.5-0.5B-Instruct (zero-shot) vs. Claude Haiku 4.5 "
            "classifying real Bugzilla bug reports as severe "
            "(blocker/critical/major) or not severe (minor/normal/trivial). "
            "Neither model is fine-tuned here."
        )

    with gr.Tab("🔴 Try it live"):
        gr.Markdown(
            "Paste a bug report below, or pick an example, and see how "
            "both models classify it side by side.",
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
                gr.Markdown("#### 🔹 Base Qwen (zero-shot)")
                with gr.Group(elem_classes="model-card"):
                    qwen_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("#### 🟣 Claude Haiku 4.5")
                with gr.Group(elem_classes="model-card"):
                    claude_out = gr.Markdown()

        run_btn.click(classify, inputs=[desc_in], outputs=[qwen_out, claude_out])

    with gr.Tab("📊 Eval dashboard"):
        missing_msg, baseline_df = build_dashboard()
        if missing_msg:
            gr.Markdown(f"⚠️ {missing_msg}")
        else:
            gr.Markdown(
                "### Real n=300 result (Claude Haiku 4.5, baseline)\n"
                "75.3% accuracy against a 77.2% majority-class baseline — "
                "did not collapse to the majority class. **Note:** this "
                "dashboard reflects the offline Claude-only experiment "
                "(see the GitHub repo's README); the base-Qwen comparison "
                "above is live-only and hasn't been run as a scored batch.",
                elem_classes="section-header",
            )
            gr.Dataframe(baseline_df, interactive=False)

if __name__ == "__main__":
    demo.launch(show_error=True)
