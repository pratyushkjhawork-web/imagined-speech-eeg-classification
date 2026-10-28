"""Dataset utilities.

Everything downstream expects ONE cached file (.npz) with:
    X       float32  (N, n_channels, n_samples)   one row per window/trial
    y       int64    (N,)                          class index 0..K-1
    subject int64    (N,)                          subject id per row
    fs      int      sampling rate (optional, default 128)
    classes str array of class names (optional)

Use scripts/prepare_kumar.py to create it from the raw download, or
make_synthetic() to smoke-test the pipeline without any real data.
"""
import numpy as np


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    s = d["subject"].astype(np.int64)
    fs = int(d["fs"]) if "fs" in d.files else 128
    classes = list(d["classes"]) if "classes" in d.files else [str(i) for i in np.unique(y)]
    return X, y, s, fs, classes


def load_full(path):
    """Like load_npz but returns a dict and includes recording ids / window start
    offsets when present (needed for recording-level metrics and temporal splits)."""
    d = np.load(path, allow_pickle=True)
    out = dict(X=d["X"].astype(np.float32), y=d["y"].astype(np.int64),
               subject=d["subject"].astype(np.int64),
               fs=int(d["fs"]) if "fs" in d.files else 128,
               classes=list(d["classes"]) if "classes" in d.files else None,
               rec=d["rec"].astype(np.int64) if "rec" in d.files else None,
               start=d["start"].astype(np.int64) if "start" in d.files else None)
    if out["classes"] is None:
        out["classes"] = [str(i) for i in np.unique(out["y"])]
    return out


def fit_channel_stats(X, eps=1e-6):
    """Channel-wise mean/std over all windows and samples. Call ONLY on training data."""
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + eps
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_channel_stats(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def subject_zscore(X, subject, eps=1e-6):
    """Per-subject channel-wise z-score (unsupervised: uses only that subject's own EEG,
    no labels, no other subjects)."""
    out = np.empty_like(X)
    for s in np.unique(subject):
        m = subject == s
        mu, sd = fit_channel_stats(X[m], eps)
        out[m] = (X[m] - mu) / sd
    return out.astype(np.float32)


def window_with_starts(sig, win, stride):
    """sig: (C, T) -> windows (n, C, win) and their start sample offsets (n,)."""
    T = sig.shape[1]
    starts = np.arange(0, T - win + 1, stride) if T >= win else np.array([], int)
    if len(starts) == 0:
        return np.empty((0, sig.shape[0], win), np.float32), starts
    return np.stack([sig[:, s:s + win] for s in starts]).astype(np.float32), starts


def zscore_per_trial(X, eps=1e-6):
    """Per-window, per-channel standardisation. Uses no cross-subject
    statistics, so it cannot leak information from the test subject."""
    m = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    return ((X - m) / (sd + eps)).astype(np.float32)


def make_synthetic(n_subjects=6, n_classes=5, per_class=40, n_channels=14,
                   n_samples=128, fs=128, seed=0):
    """Fake EEG-like data with a weak class-dependent sinusoid. ONLY for
    checking that the code runs. Results on this mean nothing scientifically."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    Xs, ys, ss = [], [], []
    for s in range(n_subjects):
        for c in range(n_classes):
            for _ in range(per_class):
                x = rng.normal(size=(n_channels, n_samples))
                x += 0.5 * np.sin(2 * np.pi * (6 + 3 * c) * t)[None, :] \
                     * rng.uniform(0.5, 1.5, size=(n_channels, 1))
                Xs.append(x); ys.append(c); ss.append(s)
    return (np.asarray(Xs, np.float32), np.asarray(ys, np.int64),
            np.asarray(ss, np.int64), fs, [f"class{i}" for i in range(n_classes)])


def make_synthetic_full(**kw):
    """make_synthetic + fake recording ids / start offsets (one recording per subject-class)."""
    X, y, s, fs, classes = make_synthetic(**kw)
    rec = np.zeros(len(X), np.int64); start = np.zeros(len(X), np.int64)
    r, k = 0, 0
    for i in range(len(X)):
        if i > 0 and (s[i] != s[i - 1] or y[i] != y[i - 1]):
            r += 1; k = 0
        rec[i] = r; start[i] = k * 16; k += 1
    return dict(X=X, y=y, subject=s, fs=fs, classes=classes, rec=rec, start=start)


def window_signal(sig, win, stride):
    """sig: (n_channels, T) -> (n_windows, n_channels, win)"""
    T = sig.shape[1]
    if T < win:
        return np.empty((0, sig.shape[0], win), np.float32)
    starts = range(0, T - win + 1, stride)
    return np.stack([sig[:, s:s + win] for s in starts]).astype(np.float32)
