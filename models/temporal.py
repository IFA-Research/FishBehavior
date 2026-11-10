import torch
import torch.nn as nn

class NoTemporal(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)

class BiLSTMTemporal(nn.Module):
    def __init__(self, input_size=640, hidden_size=256, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        if use_attention:
            self.attn = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)  # (B, T, hidden*2)
        if self.use_attention:
            scores = self.attn(x).squeeze(-1)  # (B, T)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
            x = (x * weights).sum(dim=1)  # weighted sum
        else:
            x = x[:, -1, :]
        return x

class TransformerTemporal(nn.Module):
    def __init__(self, input_size=640, num_layers=1, num_heads=8, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer(x)
        return x[:, -1, :]

class BiLSTMTransformerTemporal(nn.Module):
    def __init__(self, input_size=640, hidden_size=256, num_layers=1, num_heads=8, dropout=0.1, use_attention=False):
        super().__init__()
        self.bilstm = BiLSTMTemporal(input_size, hidden_size, use_attention=use_attention)
        self.proj = nn.Linear(hidden_size * 2, input_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Keep full sequence for transformer; attention is applied inside BiLSTM if enabled
        x, _ = self.bilstm.lstm1(x)
        x, _ = self.bilstm.lstm2(x)
        if self.bilstm.use_attention:
            scores = self.bilstm.attn(x).squeeze(-1)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            # For hybrid we still pass full sequence after optional weighting
            x = x * weights
        x = self.proj(x)
        x = self.transformer(x)
        return x[:, -1, :]

def get_temporal_module(temporal_type, use_attention=False, **kwargs):
    if temporal_type in ["none", "mean"]:
        return NoTemporal()
    elif temporal_type == "bilstm":
        # BiLSTMTemporal does not accept num_layers / num_heads
        bilstm_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ("input_size", "hidden_size")
        }
        return BiLSTMTemporal(use_attention=use_attention, **bilstm_kwargs)
    elif temporal_type == "transformer":
        # TransformerTemporal does not accept hidden_size
        transformer_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ("input_size", "num_layers", "num_heads", "dropout")
        }
        return TransformerTemporal(**transformer_kwargs)
    elif temporal_type == "bilstm_transformer":
        return BiLSTMTransformerTemporal(use_attention=use_attention, **kwargs)
    else:
        raise ValueError(f"Unknown temporal_type: {temporal_type}")
