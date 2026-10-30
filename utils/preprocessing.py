"""
Preprocessing utilities for the GRU NMT model.
Matches the exact tokenization and encoding used during training.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import streamlit as st

# Special token IDs — must match training constants
PAD, UNK, SOS, EOS = 0, 1, 2, 3
MAX_LEN = 24  # word-level sequence limit used during training


def tokenize(text: str) -> List[str]:
    """
    Lowercase + whitespace tokenization — identical to the notebook.

    Args:
        text: raw English sentence.
    Returns:
        List of lower-cased word tokens.
    """
    return text.lower().strip().split()


def encode(text: str, stoi: Dict[str, int], max_len: int = MAX_LEN) -> List[int]:
    """
    Encode text to token-id list with SOS/EOS padding.

    Follows the notebook exactly:
        ids = [SOS] + [stoi.get(w, UNK) for w in tokenize(text)[:max_len]] + [EOS]

    Args:
        text:    input sentence.
        stoi:    token → index mapping.
        max_len: maximum number of word tokens (before SOS/EOS are added).
    Returns:
        List of integer token ids.
    """
    tokens = tokenize(text)[:max_len]
    return [SOS] + [stoi.get(tok, UNK) for tok in tokens] + [EOS]


def decode_ids(ids: List[int], itos: List[str]) -> str:
    """
    Convert token ids back to a string, skipping special tokens.

    Args:
        ids:  list of integer token ids.
        itos: index → token mapping (list).
    Returns:
        Space-joined string of tokens.
    """
    words = []
    for idx in ids:
        if idx in (PAD, SOS, EOS):
            continue
        if 0 <= idx < len(itos):
            words.append(itos[idx])
        else:
            words.append("<unk>")
    return " ".join(words)


@st.cache_resource(show_spinner=False)
def load_vocabularies(vocab_path: str) -> Tuple[Dict, List, Dict, List]:
    """
    Load src/tgt vocabularies from pickle file, cached per session.

    Args:
        vocab_path: path to vocabularies.pkl
    Returns:
        (src_stoi, src_itos, tgt_stoi, tgt_itos)
    Raises:
        FileNotFoundError: if file does not exist.
    """
    path = Path(vocab_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Vocabularies file not found: {vocab_path}\n"
            "Place vocabularies.pkl in the artifacts/ directory."
        )
    with open(path, "rb") as f:
        vocabs = pickle.load(f)

    src_stoi = vocabs["src_stoi"]
    src_itos = vocabs["src_itos"]
    tgt_stoi = vocabs["tgt_stoi"]
    tgt_itos = vocabs["tgt_itos"]
    return src_stoi, src_itos, tgt_stoi, tgt_itos


def get_source_tokens(text: str) -> List[str]:
    """
    Return the token sequence that the encoder sees (including <sos>).
    Matches notebook: ['<sos>'] + tokenize(text)[:MAX_LEN]
    """
    return ["<sos>"] + tokenize(text)[:MAX_LEN]
