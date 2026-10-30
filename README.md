# 🌐 English → Telugu Neural Machine Translation

> A research demonstration comparing a **GRU Encoder-Decoder with Bahdanau Attention** (trained from scratch) against **Facebook NLLB-200 Distilled 600M** (pretrained multilingual model) for English-to-Telugu translation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)

---

## 📸 Screenshots

> _Launch the app to see the live interface._

---

## 🗂️ Project Structure

```
Neural-Machine-Translation/
│
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .gitignore
│
├── nmt_artifacts/                 # Trained model & evaluation data
│   ├── gru_bahdanau.pt            # GRU checkpoint (~63 MB)
│   ├── vocabularies.pkl           # Word-level vocabularies
│   ├── evaluation_metrics.csv     # BLEU & chrF results
│   └── sample_predictions.json   # 300 test-set translations
│
├── models/
│   ├── __init__.py
│   ├── gru_attention.py           # Encoder, BahdanauAttention, AttnDecoder, Seq2Seq
│   └── nllb_model.py              # NLLB-200 lazy loader & inference wrapper
│
└── utils/
    ├── __init__.py
    ├── preprocessing.py           # tokenize / encode (matches training exactly)
    └── visualization.py           # Attention heatmaps & metric charts
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or later
- (Optional) CUDA-capable GPU for faster inference

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Neural-Machine-Translation.git
cd Neural-Machine-Translation

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (GPU only) Install PyTorch with CUDA — see https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5. Launch the app
streamlit run app.py
```

> **NLLB-200 first run:** The model (~1.2 GB) will be downloaded automatically from HuggingFace Hub on the first translation. An internet connection is required.

---

## 📦 Dataset

**Samanantar** (ai4bharat) — the largest publicly available English-Indic parallel corpus.

| Property | Value |
|---|---|
| Language pair | English → Telugu |
| Pairs used (train) | ~27,000 |
| Pairs used (eval/test) | 1,500 each |
| Tokenization | Lowercase whitespace split |
| Sequence limit | 24 tokens |
| Min word frequency | 2 |

**Source:** [ai4bharat/samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) · CC0 / Open

---

## 🤖 Models

### Model A — GRU + Bahdanau Attention (From Scratch)

| Property | Value |
|---|---|
| Architecture | Encoder-Decoder GRU |
| Embedding dim | 256 |
| Hidden dim | 256 |
| Attention | Bahdanau (additive) |
| Source vocab | 12,093 tokens |
| Target vocab | 15,815 tokens |
| Training epochs | 3 |
| Optimizer | Adam (lr=1e-3) |
| Dataset | Samanantar-te (~27k pairs) |
| Parameters | ~14.7M |

### Model B — NLLB-200 Distilled 600M

| Property | Value |
|---|---|
| Model ID | `facebook/nllb-200-distilled-600M` |
| Architecture | Transformer (encoder-decoder) |
| Parameters | ~600M (distilled) |
| Languages | 200 |
| Source lang code | `eng_Latn` |
| Target lang code | `tel_Telu` |
| Decoding | Beam search (4 beams) |
| Fine-tuned | No (zero-shot) |

---

## 📊 Evaluation Results

Evaluated on **300 test-set sentence pairs** from Samanantar:

| Model | BLEU | chrF |
|---|---|---|
| GRU + Bahdanau (scratch) | 0.14 | 4.87 |
| NLLB-200 Distilled 600M | 8.28 | 40.37 |

> **Interpretation:** The scratch GRU model is an educational baseline — 3 epochs of training on 27k pairs with a word-level vocabulary is insufficient for quality translation. NLLB-200 benefits from massive multilingual pretraining. The chrF gap (≈35 points) highlights the advantage of subword tokenization and scale.

---

## 🖥️ Application Features

| Tab | Description |
|---|---|
| 🔤 Translate | Real-time translation with GRU, NLLB-200, or both side-by-side |
| 📊 Evaluation | BLEU & chrF scores, interactive comparison charts |
| 🔭 Attention | Bahdanau attention heatmap for GRU model |
| 📖 Examples | Browse 300 test-set predictions with reference translations |
| 📘 About | Architecture explanations, metric definitions, dataset info |

---

## ⚠️ Limitations

- **GRU model:** Word-level vocabulary causes many `<unk>` tokens for OOV words. Low BLEU is expected — this is an experimental baseline, not production-quality.
- **NLLB-200:** Not fine-tuned on this specific domain; quality depends on domain overlap with pretraining data.
- **Max length:** GRU tokenizer truncates inputs to 24 words; longer sentences will be silently truncated.
- **Vocabulary coverage:** Words appearing fewer than 2× in training are mapped to `<unk>`.

---

## 📜 License & Attribution

| Asset | License |
|---|---|
| NLLB-200 model | CC-BY-NC 4.0 (Meta AI) |
| Samanantar dataset | CC0 / Open (AI4Bharat) |
| Project code | MIT |

---

## 🙏 Acknowledgements

- [AI4Bharat](https://ai4bharat.org/) for the Samanantar corpus
- [Meta AI](https://ai.meta.com/) for the NLLB-200 model family
- [HuggingFace](https://huggingface.co/) for the Transformers library
- Bahdanau et al. (2015) — _Neural Machine Translation by Jointly Learning to Align and Translate_
