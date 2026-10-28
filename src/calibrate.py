"""LOSO + subject calibration (personalisation), clearly separated from generic LOSO.

For every held-out subject:
  1. train the GENERIC model on the other subjects (same code as src/train.py)
  2. split the held-out subject's windows into calibration / test (--cal_split)
  3. report four numbers on the SAME test windows:
       generic          : generic model, no calibration              (subject-independent)
       calibrated       : generic model fine-tuned on calibration windows with TRUE labels
       permuted control : same fine-tuning but class labels cyclically relabelled
  Interpretation: with one recording per subject/class, a model can classify windows by
  recording identity (drift, impedance, session) without any imagined-speech information.
  Class labels are arbitrary symbols for such a model, so cyclically RELABELLING the classes
  should give about the same calibrated accuracy. If calibrated ~ permuted control, the
  calibration accuracy says nothing about imagined speech: it is recording identity.
  (A torch-free LDA preview gave identical numbers by construction.) Only the generic
  LOSO number (no data from the test subject) can speak to imagined-speech decoding.

--cal_split temporal : calibration = first part of each recording; test = windows starting
                       >= one window length after it (no shared samples). Default.
--cal_split random   : paper-style random 20% of windows per recording. LEAKY (neighbouring
                       and overlapping windows land on both sides). Use only for comparison.

Usage:
  python -m src.calibrate --data data/char_fdf_w32.npz --model cnn_bilstm --norm train \
      --sched plateau --monitor loss --clip 1.0 --batch_size 256 --epochs 40 --patience 8
"""
import argparse, copy, json, os, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn

from .train import (add_train_args, DEFAULT_DROPOUT, set_seed, to_loader, evaluate,
                    train_fold, load_data)
from .models import count_params
from .protocols import calibration_split


def finetune(model, X, y, device, epochs, lr, wd, bs, clip):
    loader = to_loader(X, y, bs, True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):                      # fixed epoch budget: no peeking at test data
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); ce(model(xb), yb).backward()
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    return model


def acc_on(model, X, y, device):
    yt, lt, _ = evaluate(model, to_loader(X, y, 512, False), device)
    return float((lt.argmax(1) == yt).mean())


def main():
    ap = add_train_args(argparse.ArgumentParser())
    ap.add_argument("--cal_split", choices=["temporal", "random"], default="temporal")
    ap.add_argument("--cal_frac", type=float, default=0.2)
    ap.add_argument("--cal_epochs", type=int, default=15)
    ap.add_argument("--cal_lr", type=float, default=5e-4)
    ap.add_argument("--cal_batch", type=int, default=64)
    args = ap.parse_args()
    if args.dropout is None:
        args.dropout = DEFAULT_DROPOUT[args.model]
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = load_data(args)
    X, y, subj, rec, start, fs = d["X"], d["y"], d["subject"], d["rec"], d["start"], d["fs"]
    assert rec is not None and start is not None, "re-run prepare_kumar.py: npz needs rec/start"
    K, win = len(np.unique(y)), X.shape[2]
    subjects = np.unique(subj)[:args.max_folds] if args.max_folds else np.unique(subj)
    print(f"device={device} model={args.model} split={args.cal_split} frac={args.cal_frac} "
          f"cal_epochs={args.cal_epochs} chance={1/K:.3f}", flush=True)

    rows, nparams, t0 = [], None, time.time()
    for s in subjects:
        set_seed(args.seed + int(s))
        model, Xf, (tr, va, te), hist, best_ep, _ = train_fold(args, X, y, subj, s, K, fs, device, rng)
        nparams = count_params(model)
        idx_te = np.where(te)[0]
        cal_idx, test_idx = calibration_split(rec, start, idx_te, args.cal_frac,
                                              args.cal_split, win, rng)
        generic_full = acc_on(model, Xf[idx_te], y[idx_te], device)
        generic_test = acc_on(model, Xf[test_idx], y[test_idx], device)
        m_cal = finetune(copy.deepcopy(model), Xf[cal_idx], y[cal_idx], device, args.cal_epochs,
                         args.cal_lr, args.wd, args.cal_batch, args.clip)
        cal_acc = acc_on(m_cal, Xf[test_idx], y[test_idx], device)
        shift = int(rng.integers(1, K))          # cyclic relabel: no class keeps its label
        yp = (y + shift) % K
        m_perm = finetune(copy.deepcopy(model), Xf[cal_idx], yp[cal_idx], device, args.cal_epochs,
                          args.cal_lr, args.wd, args.cal_batch, args.clip)
        perm_acc = acc_on(m_perm, Xf[test_idx], yp[test_idx], device)
        rows.append(dict(test_subject=int(s), generic_full=generic_full, generic_on_test_part=generic_test,
                         calibrated=cal_acc, permuted_control=perm_acc,
                         n_cal=len(cal_idx), n_test=len(test_idx)))
        print(f"subj {s}: generic {generic_full:.3f} | on test part {generic_test:.3f} | "
              f"calibrated {cal_acc:.3f} | permuted-label control {perm_acc:.3f} "
              f"(cal {len(cal_idx)}, test {len(test_idx)}; {(time.time()-t0)/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    dataset = "synthetic" if args.synthetic else os.path.splitext(os.path.basename(args.data))[0]
    tag = f"{dataset}_{args.model}_seed{args.seed}_calib_{args.cal_split}"
    df.to_csv(f"{args.out}/{tag}_per_subject.csv", index=False)
    summ = {c: (float(df[c].mean()), float(df[c].std(ddof=1)) if len(df) > 1 else 0.0)
            for c in ["generic_full", "generic_on_test_part", "calibrated", "permuted_control"]}
    json.dump(dict(dataset=dataset, model=args.model, params=nparams, chance=1 / K, split=args.cal_split,
                   leaky=args.cal_split == "random", mean_std=summ, args=vars(args)),
              open(f"{args.out}/{tag}_summary.json", "w"), indent=2)
    print(f"\n=== LOSO + calibration ({args.cal_split} split) mean ± std over subjects; chance={1/K:.3f} ===")
    for c, (mu, sd) in summ.items():
        print(f"{c:22s} {mu:.4f} ± {sd:.4f}")
    if args.cal_split == "random":
        print("WARNING: random split is leakage-prone (overlapping/neighbouring windows, same recording).")
    print("Read: calibrated ~ permuted_control => calibration accuracy is explained by recording identity, "
          "not imagined-speech content. Only the generic (no-calibration) number is subject-independent evidence.")


if __name__ == "__main__":
    main()
