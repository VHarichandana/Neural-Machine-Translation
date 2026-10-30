"""
Full end-to-end validation script.
Tests all modules can be imported, model loads, and inference runs.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

print("=== 1. Vocabularies ===")
from utils.preprocessing import load_vocabularies, tokenize, encode, get_source_tokens
src_stoi, src_itos, tgt_stoi, tgt_itos = load_vocabularies('nmt_artifacts/vocabularies.pkl')
print(f"  src vocab: {len(src_stoi)}, tgt vocab: {len(tgt_stoi)}")

print("\n=== 2. Tokenize/encode ===")
sentence = "The police arrested the suspect near the market."
tokens = tokenize(sentence)
ids = encode(sentence, src_stoi)
src_toks = get_source_tokens(sentence)
print(f"  tokens: {tokens}")
print(f"  ids:    {ids}")
print(f"  src_tokens (with <sos>): {src_toks}")

print("\n=== 3. Model load ===")
import torch
from models.gru_attention import load_gru_model
device = torch.device('cpu')
model = load_gru_model('nmt_artifacts/gru_bahdanau.pt', len(src_itos), len(tgt_itos), device)
params = sum(p.numel() for p in model.parameters())
print(f"  Model: {type(model).__name__}, Parameters: {params:,}")

print("\n=== 4. Inference ===")
import numpy as np
import time

src_tensor = torch.tensor([ids])
t0 = time.perf_counter()
with torch.inference_mode():
    enc_out, hidden = model.encoder(src_tensor)
    mask = src_tensor.ne(0)
    token = torch.tensor([2])  # SOS
    out_ids, attn_rows = [], []
    for _ in range(50):
        logits, hidden, weights = model.decoder.forward_step(token, hidden, enc_out, mask)
        nxt = logits.argmax(-1)
        idx = int(nxt.item())
        attn_rows.append(weights.squeeze(0).cpu().numpy())
        if idx == 3: break  # EOS
        out_ids.append(idx)
        token = nxt
elapsed = time.perf_counter() - t0

words = [tgt_itos[i] if 0 <= i < len(tgt_itos) else '<unk>' for i in out_ids]
translation = ' '.join(words) if words else '<empty>'
attention = np.array(attn_rows)
print(f"  EN:      {sentence}")
print(f"  TE:      {translation}")
print(f"  Elapsed: {elapsed*1000:.1f} ms")
print(f"  Attn shape: {attention.shape}")

print("\n=== 5. Visualization (matplotlib) ===")
import matplotlib
matplotlib.use('Agg')
from utils.visualization import plot_attention_heatmap, plot_metric_comparison, plot_score_radar
import pandas as pd

att = attention[:len(words), :len(src_toks)]
if att.size > 0:
    fig = plot_attention_heatmap(att, src_toks[:att.shape[1]], words, title="Test Attention")
    print(f"  Attention fig created: {fig.get_size_inches()}")
else:
    print("  No output tokens -- attention matrix empty")

metrics_df = pd.read_csv('nmt_artifacts/evaluation_metrics.csv')
fig2 = plot_metric_comparison(metrics_df)
fig3 = plot_score_radar(metrics_df)
print(f"  Comparison fig: size={fig2.get_size_inches()}")
print(f"  Radar fig: size={fig3.get_size_inches()}")

print("\n=== 6. Sample predictions ===")
import json
with open('nmt_artifacts/sample_predictions.json', encoding='utf-8') as f:
    samples = json.load(f)
print(f"  {len(samples)} samples loaded")
print(f"  Keys: {list(samples[0].keys())}")
print(f"  Example EN={samples[5]['english'][:60]}")

print("\n" + "="*50)
print("ALL CHECKS PASSED")
