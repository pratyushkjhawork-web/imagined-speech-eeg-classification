"""Leave-One-Subject-Out (LOSO) training / evaluation for
   eegnet | eegnet_attn | cnn_bilstm

Baseline (unchanged behaviour):
    python -m src.train --data data/char_basic_w128.npz --model eegnet --seed 1
New pipeline (short windows, train-only normalisation, BiLSTM):
    python -m src.train --data data/char_fdf_w32.npz --model cnn_bilstm --norm train \
        --sched plateau --monitor loss --clip 1.0 --batch_size 256 --epochs 40 --patience 8
Smoke test:
    python -m src.train --synthetic --model cnn_bilstm --epochs 2 --max_folds 2 --norm train
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from scipy.special import softmax
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

from .data import (load_full, make_synthetic_full, zscore_per_trial, subject_zscore,
                   fit_channel_stats, apply_channel_stats)
from .models import build_model, count_params
from .protocols import make_split, rec_level_accuracy

DEFAULT_DROPOUT = {"eegnet": 0.5, "eegnet_attn": 0.5, "cnn_bilstm": 0.3}


def add_train_args(ap):
    ap.add_argument("--data", type=str, default="data/kumar_char.npz")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", choices=list(DEFAULT_DROPOUT), required=True)
    ap.add_argument("--norm", choices=["trial", "train", "subject", "none"], default="trial",
                    help="trial: per-window z-score (old default) | train: channel stats from TRAINING "
                         "subjects only | subject: per-subject z-score (unsupervised) | none")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=None, help="default depends on model")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--sched", choices=["cosine", "plateau"], default="cosine")
    ap.add_argument("--lr_patience", type=int, default=3, help="plateau scheduler patience")
    ap.add_argument("--monitor", choices=["acc", "loss"], default="acc",
                    help="what early stopping / checkpoint selection watches (validation subjects)")
    ap.add_argument("--clip", type=float, default=0.0, help="grad-norm clip (0=off; use 1.0 for LSTM)")
    ap.add_argument("--val_subjects", type=int, default=2)
    ap.add_argument("--train_eval_n", type=int, default=4000,
                    help="windows used to report train accuracy/loss each epoch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_folds", type=int, default=0, help="debug: limit LOSO folds")
    ap.add_argument("--out", type=str, default="results")
    return ap


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_loader(X, y, bs, shuffle):
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X)).unsqueeze(1), torch.from_numpy(y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    logits, ys, tot, n = [], [], 0.0, 0
    for xb, yb in loader:
        out = model(xb.to(device))
        if loss_fn is not None:
            tot += loss_fn(out, yb.to(device)).item() * len(yb)
        n += len(yb)
        logits.append(out.float().cpu().numpy()); ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(logits), (tot / n if loss_fn is not None else None)


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y, p), precision=pr, recall=rc, f1=f1)


def fit_model(args, model, tr_loader, va_loader, tr_eval_loader, device):
    """Train with early stopping on VALIDATION SUBJECTS only. Returns best model + history."""
    ce = nn.CrossEntropyLoss()
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.sched == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                           patience=args.lr_patience)
    best_score, best_state, best_ep, bad, hist = -np.inf, None, 0, 0, []
    for ep in range(args.epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            train_loss_fn(model(xb), yb).backward()
            if args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
        yt, lt, tl = evaluate(model, tr_eval_loader, device, ce)
        yv, lv, vl = evaluate(model, va_loader, device, ce)
        tacc, vacc = float((lt.argmax(1) == yt).mean()), float((lv.argmax(1) == yv).mean())
        if args.sched == "cosine": sched.step()
        else: sched.step(vl)
        hist.append(dict(epoch=ep + 1, train_loss=tl, train_acc=tacc, val_loss=vl, val_acc=vacc,
                         lr=opt.param_groups[0]["lr"]))
        score = vacc if args.monitor == "acc" else -vl
        if score > best_score + 1e-9:
            best_score, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, hist, best_ep


def train_fold(args, X, y, subj, test_subj, n_classes, fs, device, rng):
    """Train the generic (subject-independent) model for one LOSO fold."""
    tr, va, te, val_subjects = make_split(subj, test_subj, args.val_subjects, rng)
    Xf = X
    if args.norm == "train":                       # statistics from TRAINING subjects only
        mu, sd = fit_channel_stats(X[tr])
        Xf = apply_channel_stats(X, mu, sd)
    tr_idx = np.where(tr)[0]
    sub = rng.choice(tr_idx, size=min(args.train_eval_n, len(tr_idx)), replace=False)
    tr_loader = to_loader(Xf[tr], y[tr], args.batch_size, True)
    tr_eval = to_loader(Xf[sub], y[sub], 512, False)
    va_loader = to_loader(Xf[va], y[va], 512, False)
    model = build_model(args.model, n_classes, X.shape[1], X.shape[2], fs,
                        dropout=args.dropout).to(device)
    model, hist, best_ep = fit_model(args, model, tr_loader, va_loader, tr_eval, device)
    return model, Xf, (tr, va, te), hist, best_ep, val_subjects


def run_fold(args, X, y, subj, rec, test_subj, n_classes, fs, device, rng):
    model, Xf, (tr, va, te), hist, best_ep, val_subjects = train_fold(
        args, X, y, subj, test_subj, n_classes, fs, device, rng)
    te_loader = to_loader(Xf[te], y[te], 512, False)
    yt, lt, tl = evaluate(model, te_loader, device, nn.CrossEntropyLoss())
    pt = lt.argmax(1)
    m = metrics(yt, pt)
    if rec is not None:
        m["rec_acc"], _ = rec_level_accuracy(rec[te], yt, softmax(lt, axis=1))
    h = hist[best_ep]
    m.update(test_subject=int(test_subj), val_subjects=str(val_subjects),
             best_val_acc=h["val_acc"], best_val_loss=h["val_loss"], test_loss=tl,
             epochs_run=len(hist), best_epoch=best_ep + 1, n_test=int(te.sum()),
             train_acc_at_ckpt=h["train_acc"], train_loss_at_ckpt=h["train_loss"],
             train_acc_last=hist[-1]["train_acc"], train_acc_max=max(x["train_acc"] for x in hist))
    return m, yt, pt, count_params(model), hist


def load_data(args):
    d = make_synthetic_full() if args.synthetic else load_full(args.data)
    X = d["X"]
    if args.norm == "trial":
        X = zscore_per_trial(X)
    elif args.norm == "subject":
        X = subject_zscore(X, d["subject"])
    d["X"] = X
    return d


def main():
    ap = add_train_args(argparse.ArgumentParser())
    args = ap.parse_args()
    if args.dropout is None:
        args.dropout = DEFAULT_DROPOUT[args.model]
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = load_data(args)
    X, y, subj, rec, fs = d["X"], d["y"], d["subject"], d["rec"], d["fs"]
    n_classes = len(np.unique(y))
    subjects = np.unique(subj)[:args.max_folds] if args.max_folds else np.unique(subj)
    print(f"device={device} model={args.model} norm={args.norm} N={len(X)} shape={X.shape[1:]} "
          f"classes={n_classes} subjects={len(np.unique(subj))} chance={1/n_classes:.3f}", flush=True)

    rows, all_y, all_p, nparams, histories = [], [], [], None, {}
    t0 = time.time()
    for s in subjects:
        set_seed(args.seed + int(s))
        m, yt, pt, nparams, hist = run_fold(args, X, y, subj, rec, s, n_classes, fs, device, rng)
        rows.append(m); all_y.append(yt); all_p.append(pt); histories[int(s)] = hist
        print(f"[{args.model}] held-out subj {s}: acc={m['accuracy']:.3f} f1={m['f1']:.3f} "
              f"rec_acc={m.get('rec_acc', float('nan')):.2f} | val acc {m['best_val_acc']:.3f} "
              f"loss {m['best_val_loss']:.3f} | train acc {m['train_acc_at_ckpt']:.3f} "
              f"[last {m['train_acc_last']:.3f}, max {m['train_acc_max']:.3f}] | "
              f"best ep {m['best_epoch']}/{m['epochs_run']}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    keys = [k for k in ["accuracy", "precision", "recall", "f1", "rec_acc"] if k in df]
    summ = {k: (float(df[k].mean()), float(df[k].std(ddof=1)) if len(df) > 1 else 0.0) for k in keys}
    os.makedirs(args.out, exist_ok=True)
    dataset = "synthetic" if args.synthetic else os.path.splitext(os.path.basename(args.data))[0]
    tag = f"{dataset}_{args.model}_seed{args.seed}"
    df.to_csv(f"{args.out}/{tag}_per_subject.csv", index=False)
    cm = confusion_matrix(np.concatenate(all_y), np.concatenate(all_p), labels=list(range(n_classes)))
    np.save(f"{args.out}/{tag}_confusion.npy", cm)
    with open(f"{args.out}/{tag}_history.json", "w") as f:
        json.dump(histories, f)
    with open(f"{args.out}/{tag}_summary.json", "w") as f:
        json.dump(dict(dataset=dataset, model=args.model, params=nparams, chance=1 / n_classes,
                       mean_std=summ, args=vars(args), minutes=(time.time() - t0) / 60), f, indent=2)
    print("\n=== LOSO summary (mean ± std over subjects) ===")
    for k, (mu, sd) in summ.items():
        print(f"{k:10s} {mu:.4f} ± {sd:.4f}" + ("   (recording-level, 10 recordings/subject)" if k == "rec_acc" else ""))
    print(f"val acc {df['best_val_acc'].mean():.4f} | val loss {df['best_val_loss'].mean():.4f} | "
          f"test loss {df['test_loss'].mean():.4f} (ln(K)={np.log(n_classes):.3f} = chance loss)")
    print(f"train acc (eval mode, mean over folds): at best ckpt {df['train_acc_at_ckpt'].mean():.4f} | "
          f"last epoch {df['train_acc_last'].mean():.4f} | max {df['train_acc_max'].mean():.4f}")
    print(f"params={nparams}  chance={1/n_classes:.3f}")


if __name__ == "__main__":
    main()
