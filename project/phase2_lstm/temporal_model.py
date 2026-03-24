"""
Bi-LSTM + Multi-Head Attention for landslide temporal forecasting.

Replaces the classical HMM with a deep learning temporal model that:
  1. Processes per-country event sequences bidirectionally (Bi-LSTM)
  2. Applies multi-head self-attention to learn which past events matter most
  3. Predicts next landslide type, occurrence probability, and multi-step forecast

References:
  - "Attention-Based Recurrent Neural Networks for Landslide Temporal Prediction" (2024)
  - "Deep Learning for Landslide Susceptibility Mapping: A Review" (2024)
  - Vaswani et al., "Attention Is All You Need" (NeurIPS 2017) — attention mechanism
  - Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997) — LSTM

Input features per time step (18-dim):
  - Landslide type (7-dim one-hot)
  - Trigger category (6-dim one-hot)
  - Landslide size (1-dim, normalized)
  - Fatality count (1-dim, log-scaled)
  - Month (2-dim, cyclical sin/cos encoding)
  - Year (1-dim, normalized)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Multi-Head Self-Attention over temporal sequence.

    Given LSTM outputs of shape (B, T, hidden_dim), computes attention weights
    that indicate which time steps are most relevant for prediction.
    This provides interpretability — we can visualize which past events
    the model focuses on when making forecasts.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x:    (B, T, hidden_dim) — LSTM output sequence
            mask: (B, T) — True for valid positions, False for padding

        Returns:
            attended: (B, T, hidden_dim) — attention-weighted output
            weights:  (B, num_heads, T, T) — attention weights (for visualization)
        """
        B, T, D = x.shape

        # Multi-head projections
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, heads, T, T)

        if mask is not None:
            # Expand mask for multi-head: (B, 1, 1, T)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_expanded, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Weighted sum of values
        attended = torch.matmul(weights, V)  # (B, heads, T, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, T, D)
        attended = self.out_proj(attended)

        # Residual connection + layer norm
        attended = self.norm(x + attended)

        return attended, weights


class BiLSTMAttentionModel(nn.Module):
    """
    Bi-LSTM + Multi-Head Attention for landslide temporal forecasting.

    Architecture:
        Input (B, T, input_dim=18)
            ↓
        Bi-LSTM (2 layers, hidden=128) → (B, T, 256)
            ↓
        Multi-Head Self-Attention (4 heads) → (B, T, 256)
            ↓
        Context vector (attention-weighted pool) → (B, 256)
            ↓
        ├─ Type classifier:  Linear(256, n_types)     → next event type
        ├─ Probability head: Linear(256, 1) + sigmoid  → occurrence probability
        └─ Forecast head:    Linear(256, n_steps × n_types) → multi-step forecast
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        n_types: int = 7,
        n_forecast_steps: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_types = n_types
        self.n_forecast_steps = n_forecast_steps
        self.lstm_out_dim = hidden_dim * 2  # Bidirectional

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Multi-Head Self-Attention
        self.attention = TemporalAttention(
            hidden_dim=self.lstm_out_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Context pooling: learnable query for weighted sum
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.lstm_out_dim))

        # Output heads
        self.type_classifier = nn.Sequential(
            nn.Linear(self.lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_types),
        )

        self.probability_head = nn.Sequential(
            nn.Linear(self.lstm_out_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.forecast_head = nn.Sequential(
            nn.Linear(self.lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_forecast_steps * n_types),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        """
        Args:
            x:       (B, T, input_dim) — sequence of event features
            lengths: (B,) — actual sequence lengths (for masking padding)

        Returns:
            type_logits:     (B, n_types) — next event type logits
            occurrence_prob: (B, 1) — occurrence probability (sigmoid applied)
            forecast_logits: (B, n_steps, n_types) — multi-step type forecasts
            attn_weights:    (B, num_heads, T, T) — attention weights
        """
        B, T, _ = x.shape

        # Create mask for padded positions
        if lengths is not None:
            mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        # Input projection
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # Bi-LSTM
        if lengths is not None:
            # Pack for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=T)
        else:
            lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim * 2)

        # Multi-Head Self-Attention
        attended, attn_weights = self.attention(lstm_out, mask)  # (B, T, lstm_out_dim)

        # Context vector: use last valid position (or attention-weighted pool)
        if lengths is not None:
            # Gather last valid timestep for each batch item
            idx = (lengths - 1).clamp(min=0).long()
            context = attended[torch.arange(B, device=x.device), idx]  # (B, lstm_out_dim)
        else:
            context = attended[:, -1, :]  # (B, lstm_out_dim)

        # Output heads
        type_logits = self.type_classifier(context)          # (B, n_types)
        occurrence_prob = torch.sigmoid(self.probability_head(context))  # (B, 1)
        forecast_raw = self.forecast_head(context)           # (B, n_steps * n_types)
        forecast_logits = forecast_raw.view(B, self.n_forecast_steps, self.n_types)

        return type_logits, occurrence_prob, forecast_logits, attn_weights

    def predict_step(self, x: torch.Tensor, lengths: torch.Tensor = None):
        """
        Convenience method for inference. Returns probabilities instead of logits.

        Returns:
            type_probs:      (B, n_types)
            occurrence_prob: (B,)
            forecast_probs:  (B, n_steps, n_types)
            attn_weights:    (B, num_heads, T, T)
        """
        self.eval()
        with torch.no_grad():
            type_logits, occ_prob, forecast_logits, attn_weights = self(x, lengths)
            type_probs = F.softmax(type_logits, dim=-1)
            forecast_probs = F.softmax(forecast_logits, dim=-1)
        return type_probs, occ_prob.squeeze(-1), forecast_probs, attn_weights
