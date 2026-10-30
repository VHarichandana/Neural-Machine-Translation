"""
Neural Machine Translation — Streamlit Application
English → Telugu: GRU + Bahdanau Attention  vs.  NLLB-200 Distilled 600M
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "nmt_artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "gru_bahdanau.pt"
VOCAB_PATH = ARTIFACTS_DIR / "vocabularies.pkl"
METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.csv"
SAMPLES_PATH = ARTIFACTS_DIR / "sample_predictions.json"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Machine Translation | English → Telugu",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & base ── */
:root {
    --bg:          #0a0f1e;
    --surface:     #0f172a;
    --surface2:    #1e293b;
    --surface3:    #263248;
    --border:      #334155;
    --primary:     #6366f1;
    --primary-glow:#818cf8;
    --accent:      #10b981;
    --accent2:     #f59e0b;
    --text:        #e2e8f0;
    --muted:       #94a3b8;
    --danger:      #ef4444;
    --radius:      12px;
    --radius-lg:   18px;
}
html, body, .stApp {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Header gradient banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 40%, #064e3b 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -20%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse, rgba(99,102,241,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--muted);
    margin: 0;
    font-weight: 400;
}

/* ── Section cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--primary-glow);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Translation output box ── */
.translation-box {
    background: linear-gradient(135deg, var(--surface2), var(--surface3));
    border: 1px solid var(--primary);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    font-size: 1.25rem;
    line-height: 1.7;
    color: #f1f5f9;
    min-height: 80px;
    font-family: 'Noto Sans Telugu', 'Inter', sans-serif;
    box-shadow: 0 0 20px rgba(99,102,241,0.1);
    word-break: break-word;
}
.translation-box.nllb {
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(16,185,129,0.1);
}

/* ── Metric chips ── */
.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.3rem 0.8rem;
    font-size: 0.8rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
}
.chip.good { border-color: var(--accent); color: var(--accent); }
.chip.warn { border-color: var(--accent2); color: var(--accent2); }

/* ── Score cards ── */
.score-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    text-align: center;
}
.score-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.score-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
.score-model {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* ── Status tags ── */
.tag {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.tag-gru { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
.tag-nllb { background: rgba(16,185,129,0.12); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.tag-warn { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--primary), #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.6rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
}
.stSelectbox > div, .stRadio > div { color: var(--text) !important; }
div[data-testid="stMarkdownContainer"] h2 { color: #c7d2fe !important; }
div[data-testid="stMarkdownContainer"] h3 { color: #a5b4fc !important; }

/* ── Info / warning banners ── */
.info-box {
    background: rgba(99,102,241,0.08);
    border-left: 3px solid var(--primary);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    color: var(--muted);
}
.warn-box {
    background: rgba(245,158,11,0.08);
    border-left: 3px solid var(--accent2);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    color: #fde68a;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--primary-glow) !important;
    border-bottom: 2px solid var(--primary) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    padding: 1.4rem !important;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Cached resource loaders ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_vocabs():
    from utils.preprocessing import load_vocabularies
    return load_vocabularies(str(VOCAB_PATH))


@st.cache_resource(show_spinner=False)
def _load_gru_model(src_vocab_size: int, tgt_vocab_size: int):
    from models.gru_attention import load_gru_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_gru_model(str(CHECKPOINT_PATH), src_vocab_size, tgt_vocab_size, device), device


@st.cache_data(show_spinner=False)
def _load_metrics():
    if not METRICS_PATH.exists():
        return None
    return pd.read_csv(METRICS_PATH)


@st.cache_data(show_spinner=False)
def _load_samples():
    if not SAMPLES_PATH.exists():
        return []
    with open(SAMPLES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─── Inference helpers ────────────────────────────────────────────────────────

def run_gru_inference(sentence: str):
    """
    Greedy autoregressive decode with the GRU + Bahdanau model.

    Returns:
        (translation: str, attention: np.ndarray | None, elapsed: float)
    """
    from utils.preprocessing import encode, get_source_tokens
    PAD_ID = 0
    EOS_ID = 3
    SOS_ID = 2
    MAX_DEC = 50

    try:
        src_stoi, src_itos, tgt_stoi, tgt_itos = _load_vocabs()
        model, device = _load_gru_model(len(src_itos), len(tgt_itos))
    except Exception as e:
        return f"⚠️ Model load error: {e}", None, 0.0

    ids = encode(sentence, src_stoi)
    src_tensor = torch.tensor([ids], device=device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        enc_out, hidden = model.encoder(src_tensor)
        mask = src_tensor.ne(PAD_ID)
        token = torch.tensor([SOS_ID], device=device)
        out_ids, attn_rows = [], []
        for _ in range(MAX_DEC):
            logits, hidden, weights = model.decoder.forward_step(
                token, hidden, enc_out, mask
            )
            nxt = logits.argmax(-1)
            idx = int(nxt.item())
            attn_rows.append(weights.squeeze(0).cpu().numpy())
            if idx == EOS_ID:
                break
            out_ids.append(idx)
            token = nxt
    elapsed = time.perf_counter() - t0

    words = [tgt_itos[i] if 0 <= i < len(tgt_itos) else "<unk>" for i in out_ids]
    translation = " ".join(words) if words else "<empty>"
    attention = np.array(attn_rows) if attn_rows else None
    return translation, attention, elapsed


def run_nllb_inference(sentence: str):
    """
    Translate with NLLB-200.
    Returns:
        (translation: str, elapsed: float)
    """
    from models.nllb_model import translate_nllb
    try:
        t0 = time.perf_counter()
        translation = translate_nllb(sentence)
        elapsed = time.perf_counter() - t0
        return translation, elapsed
    except RuntimeError as e:
        return f"⚠️ {e}", 0.0


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;margin-bottom:1.5rem">
              <div style="font-size:2.2rem;margin-bottom:0.3rem">🌐</div>
              <div style="font-size:1.15rem;font-weight:700;color:#818cf8">NMT Dashboard</div>
              <div style="font-size:0.78rem;color:#64748b">English → Telugu</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🤖 Model Selection")
        model_choice = st.radio(
            "Choose model",
            ["GRU + Bahdanau Attention", "NLLB-200 (Distilled 600M)", "⚖️ Compare Both"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Device info
        device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        device_icon = "⚡" if torch.cuda.is_available() else "🖥️"
        st.markdown(
            f"""
            <div style="margin-bottom:1rem">
              <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">Device</div>
              <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.88rem;color:#e2e8f0">
                {device_icon} {device_name}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if torch.cuda.is_available():
            st.markdown(
                f'<div style="font-size:0.75rem;color:#64748b;margin-bottom:1rem">'
                f'{torch.cuda.get_device_name(0)}</div>',
                unsafe_allow_html=True,
            )

        # Model info box
        if "GRU" in model_choice:
            st.markdown(
                """
                <div class="info-box">
                  <b>GRU + Bahdanau Attention</b><br>
                  Trained from scratch on Samanantar (Telugu)<br>
                  Embedding: 256 · Hidden: 256 · Word-level vocab
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif "NLLB" in model_choice:
            st.markdown(
                """
                <div class="info-box">
                  <b>facebook/nllb-200-distilled-600M</b><br>
                  Pretrained on 200 languages × billions of pairs.<br>
                  ~600M parameters · Beam search (4 beams)
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="info-box">
                  <b>Side-by-side comparison</b><br>
                  Both models run on the same input.<br>
                  Latency shown per model.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.7rem;color:#475569;text-align:center">'
            'NMT Project · PyTorch + HuggingFace<br>'
            'Samanantar Dataset · ai4bharat'
            '</div>',
            unsafe_allow_html=True,
        )

    return model_choice


# ─── Tabs ─────────────────────────────────────────────────────────────────────

# Model display labels → internal keys
_MODEL_OPTIONS = {
    "🤖  GRU + Bahdanau Attention  (Scratch Model)": "gru",
    "🌐  NLLB-200  (Pretrained · facebook/nllb-200-distilled-600M)": "nllb",
}


def tab_translate(model_choice: str):
    st.markdown("## 🔤 Translation Workspace")

    # ── In-tab model selector ─────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-bottom:0.3rem;font-size:0.8rem;font-weight:600;'
        'color:#94a3b8;text-transform:uppercase;letter-spacing:0.07em">'
        'Select Model</div>',
        unsafe_allow_html=True,
    )
    tab_model_label = st.selectbox(
        "Select model",
        options=list(_MODEL_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
        key="tab_model_select",
    )
    tab_model_key = _MODEL_OPTIONS[tab_model_label]  # "gru" or "nllb"

    # Inline model description strip
    if tab_model_key == "gru":
        st.markdown(
            '<div class="info-box" style="margin-top:0.4rem;margin-bottom:0.8rem">'
            '<b>GRU + Bahdanau Attention</b> &nbsp;·&nbsp; '
            'Trained from scratch on Samanantar &nbsp;·&nbsp; '
            'emb=256, hidden=256, word-level vocab (12 093 src / 15 815 tgt tokens)'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-box" style="margin-top:0.4rem;margin-bottom:0.8rem;'
            'border-left-color:#10b981">'
            '<b>NLLB-200 Distilled 600M</b> &nbsp;·&nbsp; '
            'facebook/nllb-200-distilled-600M &nbsp;·&nbsp; '
            'eng_Latn &rarr; tel_Telu &nbsp;·&nbsp; Beam search (4 beams) &nbsp;·&nbsp; '
            '<span style="color:#fbbf24">First run downloads ~1.2 GB</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # ── Input area + tips ─────────────────────────────────────────────────────
    input_col, _, info_col = st.columns([3, 0.1, 1])
    with input_col:
        user_input = st.text_area(
            "English Input",
            placeholder="Enter an English sentence to translate into Telugu...",
            height=130,
            key="translate_input",
        )
    with info_col:
        tip_extra = (
            "• Keep sentences under 24 words (GRU truncates longer input)"
            if tab_model_key == "gru"
            else "• NLLB handles longer sentences well (max 128 tokens)"
        )
        st.markdown(
            f"""
            <div style="margin-top:1.6rem;font-size:0.8rem;color:#64748b">
              <b style="color:#94a3b8">Tips</b><br>
              {tip_extra}<br>
              • Results may vary &mdash; this is a research demo
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Translate button ──────────────────────────────────────────────────────
    translate_btn = st.button("🚀 Translate", type="primary", use_container_width=False)

    if translate_btn and user_input.strip():
        _do_translation(user_input.strip(), tab_model_key)
    elif translate_btn:
        st.warning("⚠️ Please enter some English text first.")


def _do_translation(sentence: str, model_key: str):
    """
    Execute translation and render results.

    Args:
        sentence:  English input string.
        model_key: "gru" or "nllb"
    """
    from utils.visualization import plot_attention_heatmap
    from utils.preprocessing import get_source_tokens

    if model_key == "gru":
        # ── GRU + Bahdanau ────────────────────────────────────────────────────
        with st.spinner("Running GRU model..."):
            translation, attention, elapsed = run_gru_inference(sentence)

        _render_translation_result(
            translation, elapsed, "GRU + Bahdanau Attention", "gru"
        )

        # Attention heatmap (GRU only)
        if attention is not None and not translation.startswith("⚠️"):
            with st.expander("🔍 Bahdanau Attention Heatmap", expanded=True):
                src_tokens = get_source_tokens(sentence)
                tgt_tokens = translation.split()
                if tgt_tokens and attention.shape[0] > 0:
                    att = attention[: len(tgt_tokens), : len(src_tokens)]
                    if att.size > 0:
                        fig = plot_attention_heatmap(
                            att,
                            src_tokens[: att.shape[1]],
                            tgt_tokens,
                            title="Bahdanau Attention Weights",
                        )
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("Attention matrix is empty for this input.")
                else:
                    st.info("No attention weights to display.")

    else:
        # ── NLLB-200 ──────────────────────────────────────────────────────────
        # Show a prominent loading notice on first use
        nllb_status = st.empty()
        nllb_status.markdown(
            '<div class="info-box" style="border-left-color:#f59e0b;color:#fde68a">'
            '⏳ <b>Loading NLLB-200...</b> The model (~1.2 GB) will be downloaded '
            'and cached on the first run. Subsequent calls are fast.'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Running NLLB-200 (downloading model on first run — please wait)..."):
            translation, elapsed = run_nllb_inference(sentence)

        # Clear the loading notice once done
        nllb_status.empty()

        _render_translation_result(
            translation, elapsed, "NLLB-200 Distilled 600M", "nllb"
        )

        if not translation.startswith("⚠️"):
            st.markdown(
                '<div class="info-box" style="margin-top:0.5rem;border-left-color:#10b981">'
                'ℹ️ NLLB-200 uses a Transformer architecture. '
                'Bahdanau attention visualization is available only for the GRU model.'
                '</div>',
                unsafe_allow_html=True,
            )


def _render_translation_result(translation: str, elapsed: float, model_name: str, model_key: str):
    """Render a single translation output card."""
    tag_cls = "tag-gru" if model_key == "gru" else "tag-nllb"
    box_cls = "" if model_key == "gru" else "nllb"
    border_col = "#6366f1" if model_key == "gru" else "#10b981"

    is_error = translation.startswith("⚠️")
    display_text = translation if not is_error else translation

    st.markdown(
        f"""
        <div style="margin-top:1rem">
          <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem">
            <span class="tag {tag_cls}">{model_name}</span>
            <span class="chip">⏱ {elapsed*1000:.0f} ms</span>
          </div>
          <div class="translation-box {box_cls}">
            {display_text if not is_error else f'<span style="color:#f87171">{display_text}</span>'}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_error:
        st.code(translation, language=None)  # copyable


def tab_evaluation():
    st.markdown("## 📊 Evaluation Dashboard")

    metrics_df = _load_metrics()
    if metrics_df is None:
        st.error("❌ evaluation_metrics.csv not found in artifacts/")
        return

    # ── Score cards ──────────────────────────────────────────────────────────
    st.markdown("### 📈 Evaluation Scores")
    cols = st.columns(4)
    score_data = [
        ("BLEU", "GRU + Bahdanau", metrics_df.loc[0, "BLEU"], "#6366f1"),
        ("BLEU", "NLLB-200", metrics_df.loc[1, "BLEU"], "#10b981"),
        ("chrF", "GRU + Bahdanau", metrics_df.loc[0, "chrF"], "#6366f1"),
        ("chrF", "NLLB-200", metrics_df.loc[1, "chrF"], "#10b981"),
    ]
    for col, (metric, model, val, color) in zip(cols, score_data):
        with col:
            st.markdown(
                f"""
                <div class="score-card">
                  <div class="score-value" style="color:{color}">{val:.2f}</div>
                  <div class="score-label">{metric}</div>
                  <div class="score-model">{model}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    from utils.visualization import plot_metric_comparison, plot_score_radar
    chart_col, radar_col = st.columns([3, 2])
    with chart_col:
        st.markdown("### 📉 Metric Comparison")
        fig = plot_metric_comparison(metrics_df)
        st.pyplot(fig, use_container_width=True)
    with radar_col:
        st.markdown("### 📋 Quick Comparison")
        fig2 = plot_score_radar(metrics_df)
        st.pyplot(fig2, use_container_width=True)

    st.markdown("---")

    # ── Metric explanations ───────────────────────────────────────────────────
    st.markdown("### 📚 Understanding the Metrics")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">📐 BLEU Score</div>
              <p style="color:#94a3b8;font-size:0.9rem">
                <b>Bilingual Evaluation Understudy</b> measures <i>n-gram precision</i>
                between the machine translation and one or more reference translations.
                A brevity penalty discourages very short outputs.<br><br>
                Scale: 0 – 100. Higher is better.<br>
                &gt;30 is considered "understandable"; &gt;50 is high quality.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol2:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">🔡 chrF Score</div>
              <p style="color:#94a3b8;font-size:0.9rem">
                <b>Character n-gram F-score</b> computes precision and recall
                at the character level, making it robust to morphology-rich
                languages like Telugu where words have complex inflections.<br><br>
                Scale: 0 – 100. Higher is better.<br>
                Often more informative than BLEU for agglutinative languages.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Data table ────────────────────────────────────────────────────────────
    st.markdown("### 🗂️ Full Results Table")
    st.dataframe(
        metrics_df.style.format({"BLEU": "{:.4f}", "chrF": "{:.4f}"}),
        use_container_width=True,
    )
    st.caption(f"Evaluated on {len(_load_samples())} sentence pairs from the Samanantar test split.")


def tab_attention():
    st.markdown("## 🔭 Attention Visualization")

    st.markdown(
        """
        <div class="info-box">
          The Bahdanau (additive) attention mechanism allows the decoder to look back
          at <i>all</i> encoder hidden states at each decoding step.
          The heatmap shows which source words the model attends to when generating each
          target word — darker / brighter cells = higher attention weight.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧪 Try a Custom Sentence")
    custom = st.text_input(
        "English sentence for attention visualization",
        value="The police arrested the suspect near the market.",
        key="attn_input",
    )
    attn_btn = st.button("🔍 Visualize Attention", key="attn_btn")

    if attn_btn and custom.strip():
        from utils.visualization import plot_attention_heatmap
        from utils.preprocessing import get_source_tokens

        with st.spinner("Running GRU model…"):
            translation, attention, elapsed = run_gru_inference(custom.strip())

        if translation.startswith("⚠️"):
            st.error(translation)
        else:
            st.markdown(
                f"""
                <div style="margin:1rem 0">
                  <span class="tag tag-gru">GRU Translation</span>
                  <div class="translation-box" style="margin-top:0.5rem">{translation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if attention is not None and attention.shape[0] > 0:
                src_tokens = get_source_tokens(custom.strip())
                tgt_tokens = translation.split()
                att = attention[: len(tgt_tokens), : len(src_tokens)]
                if att.size > 0:
                    fig = plot_attention_heatmap(
                        att,
                        src_tokens[: att.shape[1]],
                        tgt_tokens,
                        title=(f"Bahdanau Attention: {custom[:40]}..." if len(custom) > 40 else f"Bahdanau Attention: {custom}"),
                    )
                    st.pyplot(fig, use_container_width=True)
                    st.caption(f"⏱ Inference time: {elapsed*1000:.0f} ms")
                else:
                    st.info("Attention matrix empty — the model may have produced a very short output.")
            else:
                st.info("No attention weights available for this output.")

    st.markdown("---")
    st.markdown("### ℹ️ NLLB-200 Attention")
    st.markdown(
        """
        <div class="warn-box">
          NLLB-200 uses a standard Transformer self-attention architecture — it does
          <b>not</b> use Bahdanau (additive) attention.
          Cross-attention weights in Transformers are spread across 16 heads and 12 layers,
          making them less intuitive to display as a simple 2-D heatmap.
          Bahdanau attention visualization is only available for the GRU model.
        </div>
        """,
        unsafe_allow_html=True,
    )


def tab_examples():
    st.markdown("## 📖 Example Translations")

    samples = _load_samples()
    if not samples:
        st.error("❌ sample_predictions.json not found in artifacts/")
        return

    st.markdown(
        f'<div class="info-box">Loaded <b>{len(samples)}</b> sample predictions from the test set.</div>',
        unsafe_allow_html=True,
    )

    # ── Selector ─────────────────────────────────────────────────────────────
    idx = st.slider(
        "Select example",
        min_value=0,
        max_value=min(len(samples) - 1, 299),
        value=0,
        step=1,
        format="Sample %d",
    )

    # ── Also quick select good NLLB examples ─────────────────────────────────
    st.markdown("**Or jump to:**")
    jump_cols = st.columns(5)
    jump_values = [0, 10, 42, 77, 130]
    for jcol, jval in zip(jump_cols, jump_values):
        if jcol.button(f"#{jval}", key=f"jump_{jval}"):
            idx = jval

    sample = samples[idx]

    st.markdown(f"---\n#### Sample #{idx}")
    grid = st.columns([1, 1])

    with grid[0]:
        # English
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">🇬🇧 English Input</div>
              <p style="font-size:1rem;color:#e2e8f0">{sample["english"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Reference
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">📌 Reference Translation</div>
              <p style="font-size:1.05rem;color:#e2e8f0;font-family:'Noto Sans Telugu','Inter',sans-serif">{sample["reference"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with grid[1]:
        # GRU output
        st.markdown(
            f"""
            <div class="card" style="border-color:#6366f1">
              <div class="card-title">🤖 GRU + Bahdanau</div>
              <p style="font-size:1.05rem;color:#e2e8f0;font-family:'Noto Sans Telugu','Inter',sans-serif">{sample.get("scratch","N/A")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # NLLB output
        st.markdown(
            f"""
            <div class="card" style="border-color:#10b981">
              <div class="card-title">🌐 NLLB-200 Distilled 600M</div>
              <p style="font-size:1.05rem;color:#e2e8f0;font-family:'Noto Sans Telugu','Inter',sans-serif">{sample.get("nllb","N/A")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Analysis note ─────────────────────────────────────────────────────────
    gru_out = sample.get("scratch", "")
    unk_count = gru_out.count("<unk>")
    if unk_count > 0:
        st.markdown(
            f'<div class="warn-box">⚠️ GRU output contains <b>{unk_count}</b> &lt;unk&gt; token(s). '
            f'This is expected — the word-level vocabulary only covers words seen ≥2× in training.</div>',
            unsafe_allow_html=True,
        )


def tab_about():
    st.markdown("## 📘 About This Project")

    st.markdown(
        """
        <div class="card">
          <div class="card-title">🎯 Project Overview</div>
          <p style="color:#94a3b8;font-size:0.92rem">
            This application demonstrates and compares two distinct approaches to
            <b>Neural Machine Translation (NMT)</b> for the English → Telugu language pair:
            a custom-trained GRU model with Bahdanau attention, and Facebook's
            state-of-the-art NLLB-200 multilingual model.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏗️ Architecture Concepts")
        concepts = {
            "Encoder-Decoder": (
                "The encoder reads the source sentence into a context representation; "
                "the decoder generates the target sentence one token at a time, "
                "conditioned on the context."
            ),
            "GRU (Gated Recurrent Unit)": (
                "A recurrent neural network cell with two gates (reset & update) "
                "that controls how much past information to retain. More efficient "
                "than LSTM while capturing long-range dependencies."
            ),
            "Bahdanau Additive Attention": (
                "At each decoder step, a learned alignment score is computed between "
                "the decoder state and *every* encoder hidden state using an additive "
                "(MLP) scoring function. A softmax over these scores produces attention "
                "weights, which form a weighted sum of encoder states (the context vector)."
            ),
            "Teacher Forcing": (
                "During training, the decoder receives the *ground-truth* previous token "
                "as input rather than its own prediction. This stabilises early training "
                "but can cause exposure bias at inference time."
            ),
            "NLLB-200": (
                "No Language Left Behind — Meta AI's massively multilingual Transformer "
                "trained on 200 languages using billions of sentence pairs. The 600M "
                "distilled variant balances quality and inference speed."
            ),
        }
        for term, desc in concepts.items():
            with st.expander(f"📌 {term}"):
                st.markdown(
                    f'<p style="color:#94a3b8;font-size:0.88rem">{desc}</p>',
                    unsafe_allow_html=True,
                )

    with col2:
        st.markdown("### 📏 Evaluation Metrics")
        st.markdown(
            """
            <div class="card">
              <div class="card-title">📐 BLEU</div>
              <p style="color:#94a3b8;font-size:0.88rem">
                Bilingual Evaluation Understudy computes geometric mean precision
                of 1–4 gram overlaps between hypothesis and reference, with a
                brevity penalty.<br>
                <code>BLEU = BP · exp(Σ wₙ · log pₙ)</code>
              </p>
            </div>
            <div class="card">
              <div class="card-title">🔡 chrF</div>
              <p style="color:#94a3b8;font-size:0.88rem">
                Character n-gram F-score measures character-level overlap, better
                suited to morphologically rich languages. Telugu is agglutinative,
                making chrF especially informative here.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📦 Dataset")
        st.markdown(
            """
            <div class="card">
              <div class="card-title">Samanantar (ai4bharat)</div>
              <p style="color:#94a3b8;font-size:0.88rem">
                Open parallel corpus for 11 Indic languages. Telugu split used
                here: 30,000 sentence pairs sampled from the training stream,
                word-level tokenization, max 24 tokens per sentence.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### ⚠️ Limitations")
    st.markdown(
        """
        <div class="warn-box">
          <b>GRU Scratch Model:</b> Trained for only 3 epochs on 30,000 pairs with a word-level
          vocabulary (min-freq 2). Unknown words (OOV) map to &lt;unk&gt;. Expect low BLEU scores —
          this is an <i>educational baseline</i>, not a production system.<br><br>
          <b>NLLB-200:</b> Pretrained and not fine-tuned on this domain. Quality depends on
          domain match with NLLB training data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📜 License & Attribution")
    st.markdown(
        """
        <div class="info-box">
          Model: NLLB-200 © Meta AI — CC-BY-NC 4.0<br>
          Dataset: Samanantar © AI4Bharat — CC0 / open<br>
          GRU Scratch Model: trained as part of this project
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    model_choice = render_sidebar()

    # Hero banner
    st.markdown(
        """
        <div class="hero-banner">
          <div class="hero-title">🌐 Neural Machine Translation</div>
          <div class="hero-subtitle">Bridging English and Telugu through Deep Learning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Artifact health check (non-blocking)
    missing = []
    for p in [CHECKPOINT_PATH, VOCAB_PATH, METRICS_PATH, SAMPLES_PATH]:
        if not p.exists():
            missing.append(p.name)
    if missing:
        st.warning(
            f"⚠️ Missing artifact(s): {', '.join(missing)}. "
            f"Place them in the `nmt_artifacts/` directory."
        )

    # Tabs
    tabs = st.tabs([
        "🔤 Translate",
        "📊 Evaluation",
        "🔭 Attention",
        "📖 Examples",
        "📘 About",
    ])
    with tabs[0]:
        tab_translate(model_choice)
    with tabs[1]:
        tab_evaluation()
    with tabs[2]:
        tab_attention()
    with tabs[3]:
        tab_examples()
    with tabs[4]:
        tab_about()


if __name__ == "__main__":
    main()
