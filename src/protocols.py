"""Evaluation-protocol helpers (numpy only, no torch) so they can be unit-tested.

Everything that decides WHO is in train / val / test lives here, in one place, so that
train.py, calibrate.py and diagnose.py all use exactly the same logic.
"""
import numpy as np


def make_split(subj, test_subj, n_val, rng):
    """LOSO split: 1 test subject, `n_val` whole validation subjects, rest = train."""
    train_subjects = [s for s in np.unique(subj) if s != test_subj]
    val_subjects = [int(s) for s in rng.choice(train_subjects, size=n_val, replace=False)]
    tr = np.isin(subj, [s for s in train_subjects if s not in val_subjects])
    va = np.isin(subj, val_subjects)
    te = subj == test_subj
    return tr, va, te, val_subjects


def calibration_split(rec, start, idx_te, frac, mode, win, rng):
    """Split the held-out subject's windows into calibration / test index arrays.

    mode='temporal': per recording, the FIRST `frac` of windows (in time) is calibration;
        test windows are only those starting at least `win` samples after the last
        calibration window ends, so no calibration and test window share a sample.
    mode='random'  : per recording, a random `frac` of windows is calibration, the rest is
        test (the paper-style protocol). Windows from the same recording interleave in
        time, so neighbouring windows (and recording identity) leak across the split.
    """
    cal, test = [], []
    for r in np.unique(rec[idx_te]):
        ii = idx_te[rec[idx_te] == r]
        ii = ii[np.argsort(start[ii])]
        n = len(ii)
        n_cal = max(1, int(round(frac * n)))
        if mode == "temporal":
            c = ii[:n_cal]
            t = ii[start[ii] >= start[c[-1]] + win]
        elif mode == "random":
            perm = rng.permutation(n)
            c, t = ii[perm[:n_cal]], ii[perm[n_cal:]]
        else:
            raise ValueError(mode)
        cal.append(c); test.append(t)
    return np.concatenate(cal), np.concatenate(test)


def rec_level_accuracy(rec, y, probs):
    """Average the softmax over all windows of each recording, then argmax.
    Returns (accuracy over recordings, number of recordings)."""
    hits = []
    for r in np.unique(rec):
        m = rec == r
        hits.append(int(probs[m].mean(0).argmax() == y[m][0]))
    return float(np.mean(hits)), len(hits)
