"""Model 1: EEGNet baseline.  Model 2: EEGNet + lightweight self-attention.

Input shape for both: (batch, 1, n_channels, n_samples)
Reference: Lawhern et al., "EEGNet: a compact convolutional neural network
for EEG-based BCIs", J. Neural Eng. 2018.
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_classes, n_channels=14, n_samples=128, fs=128,
                 F1=8, D=2, F2=16, dropout=0.5, pool2=4):
        super().__init__()
        k1 = fs // 2  # temporal kernel = half the sampling rate (EEGNet paper)
        self.F2 = F2
        # Block 1: temporal conv -> depthwise spatial conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # Block 2: separable conv (depthwise temporal + pointwise)
        k2 = 16
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, k2), padding=(0, k2 // 2),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            self.eval()
            feat = self._features(torch.zeros(1, 1, n_channels, n_samples))
            self.train()
        self.n_tokens = feat.shape[-1]
        self.classifier = nn.Linear(F2 * self.n_tokens, n_classes)

    def _features(self, x):
        return self.block2(self.block1(x))  # (B, F2, 1, T')

    def forward(self, x):
        z = self._features(x).flatten(1)
        return self.classifier(z)


class EEGNetAttention(EEGNet):
    """EEGNet with a single lightweight multi-head self-attention layer
    applied over the temporal tokens produced by block 2."""

    def __init__(self, n_classes, n_heads=4, attn_dropout=0.25, **kwargs):
        super().__init__(n_classes, **kwargs)
        d = self.F2
        assert d % n_heads == 0, "F2 must be divisible by n_heads"
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=attn_dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        f = self._features(x).squeeze(2).transpose(1, 2)  # (B, T', F2)
        q = f + self.pos
        a, _ = self.attn(q, q, q, need_weights=False)
        f = self.norm(f + a)                              # residual + norm
        return self.classifier(f.flatten(1))


class CNNBiLSTM(nn.Module):
    """CNN feature extractor -> BiLSTM -> dense head, for SHORT multichannel EEG windows.

    Design reference: the CNN + BiLSTM family used on this dataset in the 2026 Sci. Rep.
    paper (re-implemented from scratch and sized for a Colab GPU; NOT a copy of their model).

    Input  (B, 1, C, T) or (B, C, T)          e.g. (B, 14, 32)
    Conv1d blocks treat the 14 channels as input features and convolve over time, so the
    first conv layer mixes channels (spatial) and every layer extracts temporal patterns.
      Conv1d(C->64,k3)+BN+ELU -> Conv1d(64->128,k3)+BN+ELU -> MaxPool(2) -> Dropout
      -> (B, T/2, 128) -> BiLSTM(hidden 64, 2 layers) -> (B, T/2, 128)
      -> mean over time -> Linear(128->64)+ELU+Dropout -> Linear(64->n_classes)
    """

    def __init__(self, n_classes, n_channels=14, n_samples=32, conv_channels=(64, 128),
                 kernel=3, pool=2, lstm_hidden=64, lstm_layers=2, dropout=0.3, dense=64, **_):
        super().__init__()
        layers, c_in = [], n_channels
        for c_out in conv_channels:
            layers += [nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
                       nn.BatchNorm1d(c_out), nn.ELU()]
            c_in = c_out
        layers += [nn.MaxPool1d(pool), nn.Dropout(dropout)]
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(c_in, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * lstm_hidden, dense), nn.ELU(),
                                  nn.Dropout(dropout), nn.Linear(dense, n_classes))

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)            # (B,1,C,T) -> (B,C,T)
        z = self.cnn(x)                 # (B, F, T')
        o, _ = self.lstm(z.transpose(1, 2))   # (B, T', 2H)
        return self.head(o.mean(dim=1))


@torch.no_grad()
def trace_shapes(model, x):
    """Forward one batch and return [(top-level module name, output shape), ...]."""
    shapes, hooks = [("input", tuple(x.shape))], []
    for name, m in model.named_children():
        def hook(mod, inp, out, n=name):
            o = out[0] if isinstance(out, tuple) else out
            shapes.append((n, tuple(o.shape)))
        hooks.append(m.register_forward_hook(hook))
    was = model.training
    model.eval(); model(x); model.train(was)
    for h in hooks: h.remove()
    return shapes


def build_model(name, n_classes, n_channels, n_samples, fs, **kw):
    cls = {"eegnet": EEGNet, "eegnet_attn": EEGNetAttention, "cnn_bilstm": CNNBiLSTM}[name]
    return cls(n_classes=n_classes, n_channels=n_channels,
               n_samples=n_samples, fs=fs, **kw)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
