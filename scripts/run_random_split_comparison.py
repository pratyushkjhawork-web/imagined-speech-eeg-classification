"""Matched random-window comparison of EEGNet and CNN-BiLSTM.

WARNING: This split is intentionally window-level and leakage-prone. With 50%-overlap
windows, adjacent windows from one recording can appear in train/validation/test. Results
must not be described as subject-independent generalization.

Example:
python scripts/run_random_split_comparison.py --data data/char_fdf_w32.npz --out results_random --seed 1
"""
import argparse, json, os, time, sys
from pathlib import Path
# Allow invocation as `python scripts/run_random_split_comparison.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy.special import softmax

from src.data import load_full, zscore_per_trial, fit_channel_stats, apply_channel_stats
from src.models import build_model, count_params
from src.train import set_seed, to_loader, evaluate, fit_model
from src.protocols import rec_level_accuracy

MODELS = ("eegnet", "cnn_bilstm")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='results_random')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--test_size', type=float, default=0.20)
    p.add_argument('--val_size', type=float, default=0.10, help='fraction of full data reserved for validation')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--dropout', type=float, default=0.3, help='same dropout setting passed to both models')
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--norm', choices=['train', 'trial', 'none'], default='train')
    p.add_argument('--synthetic', action='store_true', help='smoke test only; not scientific results')
    return p.parse_args()


def macro_metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
    return {'accuracy': float(accuracy_score(y, pred)), 'precision': float(p), 'recall': float(r), 'f1': float(f)}


def main():
    args = parse_args()
    if args.synthetic:
        from src.data import make_synthetic_full
        d = make_synthetic_full(n_subjects=6, n_classes=5, per_class=40, n_samples=32, seed=args.seed)
    else:
        d = load_full(args.data)
    X, y = d['X'], d['y']
    rec = d.get('rec')
    n_classes = len(np.unique(y))
    idx = np.arange(len(y))
    # Fixed, stratified index split shared by both models.
    idx_trainval, idx_test = train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=y)
    val_fraction_of_trainval = args.val_size / (1.0 - args.test_size)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=val_fraction_of_trainval,
                                          random_state=args.seed + 1, stratify=y[idx_trainval])

    if args.norm == 'trial':
        X = zscore_per_trial(X)
        Xf = X
    elif args.norm == 'train':
        mu, sd = fit_channel_stats(X[idx_train])
        Xf = apply_channel_stats(X, mu, sd)
    else:
        Xf = X.astype(np.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    dataset = 'synthetic' if args.synthetic else os.path.splitext(os.path.basename(args.data))[0]
    print(f'device={device}; dataset={dataset}; N={len(y)}; shape={X.shape[1:]}; classes={n_classes}; '
          f'train/val/test={len(idx_train)}/{len(idx_val)}/{len(idx_test)}; chance={1/n_classes:.3f}')
    rows, all_preds = [], {}
    start_time = time.time()
    for model_name in MODELS:
        # Reset seed so initialization/shuffle randomness starts consistently for each architecture.
        set_seed(args.seed)
        rng = np.random.default_rng(args.seed + 11)
        tr_loader = to_loader(Xf[idx_train], y[idx_train], args.batch_size, True)
        va_loader = to_loader(Xf[idx_val], y[idx_val], args.batch_size, False)
        # Training diagnostic uses all train windows; this does not affect gradient updates.
        tr_eval = to_loader(Xf[idx_train], y[idx_train], min(512, len(idx_train)), False)
        model = build_model(model_name, n_classes, X.shape[1], X.shape[2], d.get('fs', 128), dropout=args.dropout).to(device)
        fit_args = argparse.Namespace(epochs=args.epochs, patience=args.patience, lr=args.lr, wd=args.wd,
            label_smoothing=0.0, sched='plateau', lr_patience=3, monitor='loss', clip=args.clip)
        model, hist, best_ep = fit_model(fit_args, model, tr_loader, va_loader, tr_eval, device)
        te_loader = to_loader(Xf[idx_test], y[idx_test], 512, False)
        yt, logits, loss = evaluate(model, te_loader, device, nn.CrossEntropyLoss())
        pred = logits.argmax(1)
        row = {'model': model_name, **macro_metrics(yt, pred), 'test_loss': float(loss),
               'params': count_params(model), 'best_epoch': int(best_ep + 1), 'epochs_run': len(hist),
               'n_train': len(idx_train), 'n_val': len(idx_val), 'n_test': len(idx_test),
               'seed': args.seed, 'split': 'stratified_random_window_80_10_20'}
        if rec is not None:
            row['recording_level_accuracy'] = rec_level_accuracy(rec[idx_test], yt, softmax(logits, axis=1))[0]
        rows.append(row)
        all_preds[model_name] = {'y_true': yt.tolist(), 'y_pred': pred.tolist(), 'test_indices': idx_test.tolist(),
                                 'history': hist}
        tag = f'{dataset}_{model_name}_random_seed{args.seed}'
        pd.DataFrame([row]).to_csv(os.path.join(args.out, tag + '_summary.csv'), index=False)
        np.save(os.path.join(args.out, tag + '_confusion.npy'), confusion_matrix(yt, pred, labels=list(range(n_classes))))
        with open(os.path.join(args.out, tag + '_details.json'), 'w') as f:
            json.dump({'summary': row, 'history': hist, 'test_indices': idx_test.tolist(),
                       'y_true': yt.tolist(), 'y_pred': pred.tolist()}, f, indent=2)
        print(f"{model_name}: acc={row['accuracy']:.4f}, macro-F1={row['f1']:.4f}, "
              f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, params={row['params']}, "
              f"best_epoch={row['best_epoch']}", flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(args.out, f'{dataset}_random_split_comparison_seed{args.seed}.csv'), index=False)
    with open(os.path.join(args.out, f'{dataset}_random_split_comparison_seed{args.seed}.json'), 'w') as f:
        json.dump({'dataset': dataset, 'seed': args.seed, 'split_indices': {'train': idx_train.tolist(), 'validation': idx_val.tolist(), 'test': idx_test.tolist()},
                   'settings': vars(args), 'models': rows,
                   'warning': 'Random window split is leakage-prone. Do not interpret as subject-independent generalization.'}, f, indent=2)
    print(f"Saved comparison to {args.out}; elapsed={(time.time()-start_time)/60:.1f} min")
    print('WARNING: same random-window split is used for both models, but overlapping windows can leak across partitions.')

if __name__ == '__main__':
    main()
