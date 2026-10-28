"""Pre-flight diagnostics. Run this BEFORE spending hours on full LOSO.

  python -m src.diagnose --data data/char_fdf_w32.npz --model cnn_bilstm --norm train
"""
import argparse, json, os
import numpy as np
import torch, torch.nn as nn

from .train import add_train_args, DEFAULT_DROPOUT, set_seed, to_loader, evaluate, load_data
from .models import build_model, count_params, trace_shapes
from .protocols import make_split
from .data import fit_channel_stats


def main():
    ap = add_train_args(argparse.ArgumentParser())
    ap.add_argument("--overfit_subjects", type=int, default=3)
    ap.add_argument("--overfit_epochs", type=int, default=60)
    args = ap.parse_args()
    if args.dropout is None:
        args.dropout = DEFAULT_DROPOUT[args.model]
    set_seed(args.seed); rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = load_data(args)
    X, y, subj, rec, fs = d["X"], d["y"], d["subject"], d["rec"], d["fs"]
    K = len(np.unique(y)); subs = np.unique(subj)
    ok = True

    print("\n[1] INPUT DIMENSIONS")
    print(f"    X {X.shape} {X.dtype} | y {y.shape} | subjects {len(subs)} | classes {K} | fs {fs} Hz")
    print(f"    window = {X.shape[2]} samples = {X.shape[2]/fs*1000:.0f} ms | norm = {args.norm}")
    print(f"    per-channel mean range [{X.mean((0,2)).min():.3f}, {X.mean((0,2)).max():.3f}], "
          f"std range [{X.std((0,2)).min():.3f}, {X.std((0,2)).max():.3f}]")

    print("[2] NaN / Inf")
    fin = bool(np.isfinite(X).all()); ok &= fin
    print(f"    all finite: {fin}")

    print("[3] LABEL BALANCE / WINDOW COUNTS")
    cc = np.bincount(y, minlength=K)
    print(f"    windows per class: {cc.tolist()}  (min/max {cc.min()/cc.max():.2f}); chance = {1/K:.3f}")
    tab = np.zeros((len(subs), K), int)
    for i, s in enumerate(subs):
        tab[i] = np.bincount(y[subj == s], minlength=K)
    print(f"    windows per subject x class: min {tab.min()}, max {tab.max()}; "
          f"every subject has all classes: {bool((tab > 0).all())}")
    if rec is not None:
        print(f"    recordings: {len(np.unique(rec))} (expected subjects x classes = {len(subs)*K}); "
              f"windows/recording ~ {len(X)//len(np.unique(rec))}")

    print("[4] SUBJECT DISJOINTNESS (all LOSO folds)")
    for s in subs:
        tr, va, te, _ = make_split(subj, s, args.val_subjects, np.random.default_rng(0))
        S = [set(np.unique(subj[m]).tolist()) for m in (tr, va, te)]
        assert not (S[0] & S[1] or S[0] & S[2] or S[1] & S[2]) and (tr | va | te).all(), f"overlap in fold {s}"
    print("    OK: train / val / test subjects are disjoint in every fold and cover all subjects")

    print("[5] NORMALISATION LEAKAGE TEST")
    tr, va, te, _ = make_split(subj, subs[0], args.val_subjects, np.random.default_rng(0))
    X2 = X.copy(); X2[te] = X2[te] * 50 + 7           # wreck the TEST subject's data
    m1, s1 = fit_channel_stats(X[tr]); m2, s2 = fit_channel_stats(X2[tr])
    same = bool(np.allclose(m1, m2) and np.allclose(s1, s2)); ok &= same
    ma, sa = fit_channel_stats(X)
    print(f"    train-only stats unchanged when test data is altered: {same} "
          f"(stats over ALL data would differ by {np.abs(sa - s1).max():.3g} in std)")

    print("[6] MODEL")
    model = build_model(args.model, K, X.shape[1], X.shape[2], fs, dropout=args.dropout).to(device)
    xb = torch.from_numpy(X[:4]).unsqueeze(1).to(device)
    print(f"    {args.model}: params={count_params(model)}  dropout={args.dropout}")
    for n, sh in trace_shapes(model, xb):
        print(f"    {n:12s} -> {sh}")
    if hasattr(model, "lstm"):
        print(f"    BiLSTM: hidden={model.lstm.hidden_size} layers={model.lstm.num_layers} bidirectional=True")
    print(f"    optimiser AdamW lr={args.lr} wd={args.wd} | sched={args.sched} | batch={args.batch_size} "
          f"| epochs<={args.epochs} patience={args.patience} monitor={args.monitor} clip={args.clip}")

    print(f"[7] CAN THE MODEL OVERFIT A SMALL SUBSET? ({args.overfit_subjects} training subjects, no dropout)")
    pool = [int(s) for s in subs if s != subs[0]][:args.overfit_subjects]
    idx = np.where(np.isin(subj, pool))[0]
    idx = rng.choice(idx, size=min(1500, len(idx)), replace=False)
    m = build_model(args.model, K, X.shape[1], X.shape[2], fs, dropout=0.0).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0); ce = nn.CrossEntropyLoss()
    loader = to_loader(X[idx], y[idx], 128, True); evl = to_loader(X[idx], y[idx], 512, False)
    final = 0.0
    for ep in range(1, args.overfit_epochs + 1):
        m.train()
        for xb_, yb_ in loader:
            xb_, yb_ = xb_.to(device), yb_.to(device)
            opt.zero_grad(); ce(m(xb_), yb_).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 10 == 0 or ep == args.overfit_epochs:
            yt, lt, _ = evaluate(m, evl, device)
            final = float((lt.argmax(1) == yt).mean())
            print(f"    epoch {ep:3d}: train acc {final:.3f} (chance {1/K:.3f})")
    print(f"    {'PASS' if final > 0.8 else 'FAIL'}: "
          "memorising 1500 windows shows the code/optimiser/capacity work. It does NOT show the "
          "signal is learnable across subjects (any net can memorise noise).")
    ok &= final > 0.8

    print("[8] PREPROCESSING SANITY (from prepare_kumar report)")
    rp = args.data.replace(".npz", "_report.json")
    if os.path.exists(rp):
        r = json.load(open(rp))
        print(f"    pipeline={r.get('pipeline')} finite={r.get('finite')}")
        for k, v in r.get("band_power_before_after", {}).items():
            print(f"    {k:12s} power {v['before']:.3g} -> {v['after']:.3g}  (x{v['after']/max(v['before'],1e-30):.3g})")
        if r.get("subjects"):
            vk = [v["variance_kept"] for v in r["subjects"].values()]
            nr = r.get("ica_components_removed_per_subject")
            print(f"    ICA removed per subject: {nr}")
            print(f"    variance kept after ICA: min {min(vk):.2f} median {np.median(vk):.2f} max {max(vk):.2f} "
                  f"(very low values = one huge artefact component or over-removal)")
    else:
        print(f"    no report found at {rp}")
    print(f"\nOVERALL: {'all automatic checks passed' if ok else 'SOME CHECKS FAILED - fix before running LOSO'}")


if __name__ == "__main__":
    main()
