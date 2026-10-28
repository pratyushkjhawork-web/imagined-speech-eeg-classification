"""Signal processing for the paper-inspired FD-F pipeline (numpy / scipy / sklearn only).

Pipeline (per category):  demean -> ICA artifact removal (per subject)
                          -> FFT zero-phase band-reject (default 4-15 Hz, cosine ramps)
                          -> demean.  Windowing + normalisation happen later.

Everything here is UNSUPERVISED and uses no labels and no data from other subjects,
so it cannot leak information between train and test subjects.
"""
import warnings
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis

# channel order in this project: AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4
FRONTAL = [0, 1, 12, 13]  # AF3, F7, F8, AF4 (eye-blink / saccade artefacts show up here)


def demean(sig):
    return sig - sig.mean(axis=1, keepdims=True)


def bandpass(sig, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=1)


def fft_band_reject(sig, fs, lo=4.0, hi=15.0, ramp=2.0):
    """Zero-phase band-reject in the frequency domain.
    Gain is 0 inside [lo, hi] and rises to 1 over cosine ramps of width `ramp` Hz."""
    n = sig.shape[1]
    F = np.fft.rfft(sig, axis=1)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    gain = np.ones_like(f)
    gain[(f >= lo) & (f <= hi)] = 0.0
    r1 = (f >= lo - ramp) & (f < lo)
    gain[r1] = 0.5 * (1 + np.cos(np.pi * (f[r1] - (lo - ramp)) / ramp))   # 1 -> 0
    r2 = (f > hi) & (f <= hi + ramp)
    gain[r2] = 0.5 * (1 - np.cos(np.pi * (f[r2] - hi) / ramp))            # 0 -> 1
    return np.fft.irfft(F * gain[None, :], n=n, axis=1)


def _robust_z(v):
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-12
    return (v - med) / mad


def ica_clean_subject(recs, fs=128, thr=2.5, max_remove=3, seed=0):
    """Fit FastICA on all recordings of ONE subject (concatenated), flag artefact
    components automatically, zero them, and return cleaned recordings + a report.

    A component is flagged when ANY of these metrics is a robust outlier
    (> thr robust-sigma) relative to the other components of the same subject:
      * excess kurtosis          (blinks / spikes are very peaky)
      * peak amplitude (in sigma)
      * share of power below 4 Hz (slow eye movements / drift)
      * share of topography weight on frontal channels (AF3/AF4/F7/F8)
    At most `max_remove` components are removed. This is a heuristic inspired by the
    paper's '2.5 sigma' rule; the paper's exact criterion is not specified.
    """
    from sklearn.decomposition import FastICA
    C = recs[0].shape[0]
    lens = [r.shape[1] for r in recs]
    X = np.concatenate(recs, axis=1).T                      # (N, C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=C, whiten="unit-variance", max_iter=1000,
                      tol=1e-3, random_state=seed)
        S = ica.fit_transform(X)                            # (N, C) sources
    A = ica.mixing_                                         # (C_channels, C_components)
    f, P = welch(S.T, fs=fs, nperseg=256, axis=-1)
    low_share = P[:, f < 4].sum(1) / P.sum(1)
    w = np.abs(A)
    front_share = w[FRONTAL].sum(0) / w.sum(0)
    metrics = {"kurtosis": kurtosis(S, axis=0), "peak_sigma": np.abs(S).max(0),
               "low_freq_share": low_share, "frontal_share": front_share}
    score = np.max([_robust_z(v) for v in metrics.values()], axis=0)
    bad = [int(k) for k in np.argsort(-score) if score[k] > thr][:max_remove]
    S2 = S.copy(); S2[:, bad] = 0.0
    Xc = ica.inverse_transform(S2).T                        # (C, N)
    out, i = [], 0
    for L in lens:
        out.append(Xc[:, i:i + L].astype(np.float64)); i += L
    var_kept = float(Xc.var() / X.var())
    report = {"removed_components": bad, "n_removed": len(bad),
              "scores": [float(s) for s in score], "variance_kept": var_kept,
              "metrics": {k: [float(x) for x in v] for k, v in metrics.items()}}
    return out, report
