"""
NLLB-200 Distilled 600M translation model wrapper.
Handles lazy loading, Streamlit caching, and graceful error handling.
"""

from __future__ import annotations

import torch
import streamlit as st

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "tel_Telu"


@st.cache_resource(show_spinner=False)
def load_nllb():
    """
    Load NLLB tokenizer and model, cached for the session.

    Returns:
        (tokenizer, model, device, telugu_id) on success.
        Raises RuntimeError with a user-friendly message on failure.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError as e:
        raise RuntimeError(
            "transformers library not installed. Run: pip install transformers sentencepiece"
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID, src_lang=SRC_LANG)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID)
        model = model.to(device)
        model.eval()
        telugu_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
        return tokenizer, model, device, telugu_id
    except Exception as e:
        raise RuntimeError(
            f"Could not load NLLB-200 model '{NLLB_MODEL_ID}'.\n"
            f"This requires an internet connection and ~1.2 GB disk space.\n"
            f"Original error: {e}"
        ) from e


def translate_nllb(text: str) -> str:
    """
    Translate a single English sentence to Telugu using NLLB-200.

    Args:
        text: English input string.
    Returns:
        Telugu translation string.
    Raises:
        RuntimeError: if model could not be loaded.
    """
    tokenizer, model, device, telugu_id = load_nllb()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=telugu_id,
            max_new_tokens=128,
            num_beams=4,
        )

    translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return translation


def nllb_model_info() -> dict:
    """Return display-friendly model information."""
    return {
        "Model ID": NLLB_MODEL_ID,
        "Parameters": "~600M",
        "Architecture": "Transformer (encoder-decoder)",
        "Training": "Pretrained on 200 languages × billions of sentence pairs",
        "Source lang": SRC_LANG,
        "Target lang": TGT_LANG,
        "Decoding": "Beam search (4 beams)",
    }
