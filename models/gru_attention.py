"""
GRU Encoder-Decoder with Bahdanau Attention
Exact architecture reproduction from the training notebook.

Architecture details:
- Embedding dim: 256
- GRU hidden dim: 256
- Bahdanau (additive) attention
- Special tokens: PAD=0, UNK=1, SOS=2, EOS=3
"""

import torch
import torch.nn as nn

# Special token IDs (must match training)
PAD, UNK, SOS, EOS = 0, 1, 2, 3


class Encoder(nn.Module):
    """Single-layer GRU encoder."""

    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T] token ids
        Returns:
            outputs: [B, T, H] encoder hidden states
            hidden:  [1, B, H] final hidden state
        """
        emb = self.embedding(x)
        outputs, hidden = self.gru(emb)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.W1 = nn.Linear(hidden, hidden, bias=False)
        self.W2 = nn.Linear(hidden, hidden, bias=False)
        self.V = nn.Linear(hidden, 1, bias=False)

    def forward(self, query: torch.Tensor, values: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            query:  [B, 1, H]  decoder hidden state (last layer, unsqueezed)
            values: [B, S, H]  encoder outputs
            mask:   [B, S]     True where token is not PAD
        Returns:
            context: [B, 1, H]
            weights: [B, S]    attention distribution
        """
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), values)
        return context, weights


class AttnDecoder(nn.Module):
    """GRU decoder with Bahdanau attention."""

    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.attn = BahdanauAttention(hidden)
        self.gru = nn.GRU(emb_dim + hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden * 2, vocab_size)

    def forward_step(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        enc_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        """
        Single decoding step.

        Args:
            token:       [B] current token ids
            hidden:      [1, B, H] decoder hidden state
            enc_outputs: [B, S, H] encoder outputs
            src_mask:    [B, S] non-padding mask
        Returns:
            logits:  [B, V]
            hidden:  [1, B, H]
            weights: [B, S]
        """
        emb = self.embedding(token.unsqueeze(1))             # [B, 1, E]
        context, weights = self.attn(
            hidden[-1].unsqueeze(1), enc_outputs, src_mask
        )                                                     # context [B, 1, H]
        out, hidden = self.gru(
            torch.cat([emb, context], dim=-1), hidden
        )                                                     # out [B, 1, H]
        logits = self.fc(
            torch.cat([out, context], dim=-1).squeeze(1)
        )                                                     # [B, V]
        return logits, hidden, weights


class Seq2Seq(nn.Module):
    """Full sequence-to-sequence model with attention."""

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        emb_dim: int = 256,
        hidden: int = 256,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab, emb_dim, hidden)
        self.decoder = AttnDecoder(tgt_vocab, emb_dim, hidden)

    def forward(self, src, tgt, teacher_forcing=0.5):
        """Training forward pass (not used for inference)."""
        import random
        enc_out, hidden = self.encoder(src)
        src_mask = src.ne(PAD)
        tgt_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(
            tgt.size(0), tgt.size(1) - 1, tgt_vocab_size, device=src.device
        )
        token = tgt[:, 0]
        for step in range(1, tgt.size(1)):
            logits, hidden, _ = self.decoder.forward_step(
                token, hidden, enc_out, src_mask
            )
            outputs[:, step - 1] = logits
            use_teacher = random.random() < teacher_forcing
            token = tgt[:, step] if use_teacher else logits.argmax(-1)
        return outputs


def load_gru_model(checkpoint_path: str, src_vocab_size: int, tgt_vocab_size: int, device: torch.device) -> Seq2Seq:
    """
    Load trained GRU model from checkpoint.

    Args:
        checkpoint_path: Path to gru_bahdanau.pt
        src_vocab_size:  Number of source tokens (from vocabularies.pkl)
        tgt_vocab_size:  Number of target tokens (from vocabularies.pkl)
        device:          torch.device to load onto
    Returns:
        Loaded model in eval mode.
    Raises:
        FileNotFoundError: if checkpoint missing
        RuntimeError: on architecture mismatch
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Seq2Seq(src_vocab_size, tgt_vocab_size, emb_dim=256, hidden=256)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model
