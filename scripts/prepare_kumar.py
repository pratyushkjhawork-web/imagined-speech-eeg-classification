"""Convert the Kumar imagined-speech EEG archive (EDF files) into data/<name>.npz.

Verified layout of the archive (Emotiv EPOC, 128 Hz):
    Imagined_speech_EEG_edf/{Char,Digit,Image}/name<SUBJECT>_<CLASS>.edf
    * one EDF = ONE recording (~12 s) of ONE subject imagining ONE class
    * the 14 EEG channels are AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4
      (the other EDF signals - counter, gyro, markers, etc. - are ignored)

Two preprocessing pipelines (--pipeline):
  basic : trim -> demean -> 0.5-40 Hz band-pass            (what the EEGNet baseline used)
  fdf   : trim -> demean -> ICA artefact removal (per subject) -> FFT zero-phase band-reject
          (default 4-15 Hz, 2 Hz cosine ramps) -> demean   (inspired by the 2026 Sci. Rep. paper)

Windows keep their subject id, recording id (`rec`) and start offset (`start`) so that
recording-level metrics and temporal calibration splits are possible.

Examples:
    python scripts/prepare_kumar.py --raw archive.zip --category Char --pipeline basic \
        --win 128 --stride 64 --out data/char_basic_w128.npz
    python scripts/prepare_kumar.py --raw archive.zip --category Char --pipeline fdf \
        --win 32 --stride 16 --out data/char_fdf_w32.npz
"""
import argparse, json, os, re, sys, zipfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import window_with_starts
from src.signal_proc import demean, bandpass, fft_band_reject, ica_clean_subject

EEG_CH = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
FS = 128


def read_edf(raw: bytes):
    """Return dict label -> 1-D float array (physical units) for one EDF file."""
    hb = int(raw[184:192]); nrec = int(raw[236:244]); ns = int(raw[252:256])
    labels = [raw[256 + i * 16:272 + i * 16].decode("latin1").strip() for i in range(ns)]
    o = 256 + ns * 16 + ns * 80 + ns * 8                      # skip transducer + dims
    def arr(o):
        return np.array([float(raw[o + i * 8:o + (i + 1) * 8]) for i in range(ns)]), o + ns * 8
    pmin, o = arr(o); pmax, o = arr(o); dmin, o = arr(o); dmax, o = arr(o)
    o += ns * 80
    spr = [int(raw[o + i * 8:o + (i + 1) * 8]) for i in range(ns)]
    assert len(set(spr)) == 1, "mixed sampling rates not supported"
    n = spr[0]
    data = np.frombuffer(raw[hb:hb + nrec * ns * n * 2], dtype="<i2")
    if data.size != nrec * ns * n:
        raise ValueError("truncated EDF data")
    data = data.reshape(nrec, ns, n).transpose(1, 0, 2).reshape(ns, nrec * n).astype(np.float64)
    gain = (pmax - pmin) / (dmax - dmin)
    phys = (data - dmin[:, None]) * gain[:, None] + pmin[:, None]
    return {l: phys[i] for i, l in enumerate(labels)}


def iter_files(raw, category):
    pat = re.compile(rf"(?:^|/){category}/name(\d+)_(.+)\.edf$", re.I)
    if os.path.isfile(raw) and raw.endswith(".zip"):
        z = zipfile.ZipFile(raw)
        for info in z.infolist():
            m = pat.search(info.filename)
            if m:
                yield int(m.group(1)), m.group(2).lower(), (lambda i=info: z.read(i))
    else:
        for root, _, fs in os.walk(raw):
            for f in fs:
                p = os.path.join(root, f).replace("\\", "/")
                m = pat.search(p)
                if m:
                    yield int(m.group(1)), m.group(2).lower(), (lambda p=p: open(p, "rb").read())


def band_power(sig, lo, hi):
    from scipy.signal import welch
    f, P = welch(sig, fs=FS, nperseg=256, axis=-1)
    return float(P[:, (f >= lo) & (f < hi)].sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="archive.zip or extracted folder")
    ap.add_argument("--category", choices=["Char", "Digit", "Image"], required=True)
    ap.add_argument("--out", default="data/kumar.npz")
    ap.add_argument("--pipeline", choices=["basic", "fdf"], default="basic")
    ap.add_argument("--win", type=int, default=128, help="window length in samples")
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--trim", type=float, default=1.0, help="seconds dropped at start AND end")
    ap.add_argument("--no_bandpass", action="store_true", help="basic pipeline only")
    ap.add_argument("--reject", type=float, nargs=2, default=[4.0, 15.0], help="fdf band-reject Hz")
    ap.add_argument("--ica_thr", type=float, default=2.5)
    ap.add_argument("--ica_max_remove", type=int, default=3)
    ap.add_argument("--keep_incomplete", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # ---- read + trim + demean ------------------------------------------------------
    recs, skipped = {}, []
    t = int(a.trim * FS)
    for sid, cls, reader in iter_files(a.raw, a.category):
        try:
            d = read_edf(reader())
            sig = np.stack([d[c] for c in EEG_CH])
        except Exception as e:
            skipped.append((sid, cls, str(e)[:40])); continue
        sig = sig[:, t:sig.shape[1] - t] if t else sig
        recs[(sid, cls)] = demean(sig)

    classes = sorted({c for _, c in recs})
    by_sub = {}
    for (s, c) in recs: by_sub.setdefault(s, set()).add(c)
    keep = [s for s, cs in sorted(by_sub.items()) if a.keep_incomplete or cs == set(classes)]
    dropped = sorted(set(by_sub) - set(keep))
    raw_recs = {k: v.copy() for k, v in recs.items()}

    # ---- preprocessing -------------------------------------------------------------
    report = {"pipeline": a.pipeline, "category": a.category, "subjects": {}}
    if a.pipeline == "basic":
        for k in recs:
            recs[k] = recs[k] if a.no_bandpass else demean(bandpass(recs[k], FS, 0.5, 40.0))
    else:
        for s in keep:
            keys = sorted(k for k in recs if k[0] == s)
            cleaned, rep = ica_clean_subject([recs[k] for k in keys], fs=FS, thr=a.ica_thr,
                                             max_remove=a.ica_max_remove, seed=a.seed)
            report["subjects"][int(s)] = rep
            for k, c in zip(keys, cleaned):
                recs[k] = demean(fft_band_reject(c, FS, a.reject[0], a.reject[1], 2.0))
        n_rem = [r["n_removed"] for r in report["subjects"].values()]
        report["ica_components_removed_per_subject"] = n_rem
        report["reject_band_hz"] = a.reject

    # ---- sanity numbers before/after (helps verify the filters did what we think) ----
    ex = sorted(k for k in recs if k[0] in keep)[:10]
    if ex:
        report["example_subject"] = int(ex[0][0])
        report["band_power_before_after"] = {
            name: {"before": float(np.mean([band_power(raw_recs[k], lo, hi) for k in ex])),
                   "after": float(np.mean([band_power(recs[k], lo, hi) for k in ex]))}
            for name, (lo, hi) in {"delta_0.5-4": (0.5, 4), "theta_4-8": (4, 8), "alpha_8-13": (8, 13),
                                   "beta_13-30": (13, 30), "gamma_30-60": (30, 60)}.items()}
        report["finite"] = bool(all(np.isfinite(recs[k]).all() for k in recs))

    # ---- windowing -----------------------------------------------------------------
    Xs, ys, ss, rr, st = [], [], [], [], []
    rec_id = 0
    for new_id, s in enumerate(keep):
        for ci, c in enumerate(classes):
            if (s, c) not in recs: continue
            w, starts = window_with_starts(recs[(s, c)].astype(np.float32), a.win, a.stride)
            Xs.append(w); ys += [ci] * len(w); ss += [new_id] * len(w)
            rr += [rec_id] * len(w); st += starts.tolist(); rec_id += 1
    X = np.concatenate(Xs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, X=X, y=np.array(ys, np.int64), subject=np.array(ss, np.int64),
                        rec=np.array(rr, np.int64), start=np.array(st, np.int64), fs=FS,
                        classes=np.array(classes), orig_subject_ids=np.array(keep),
                        pipeline=a.pipeline, win=a.win, stride=a.stride)
    report.update(win=a.win, stride=a.stride, n_windows=int(len(X)), n_subjects=len(keep),
                  n_recordings=rec_id, windows_per_recording=int(len(X) // max(rec_id, 1)))
    rp = a.out.replace(".npz", "_report.json")
    json.dump(report, open(rp, "w"), indent=1)
    try:  # PSD before/after plot for one subject
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from scipy.signal import welch
        s0 = ex[0][0]; ks = [k for k in recs if k[0] == s0]
        def psd(d):
            f, P = welch(np.concatenate([d[k] for k in ks], axis=1), fs=FS, nperseg=256, axis=-1)
            return f, P.mean(0)
        f, p0 = psd(raw_recs); _, p1 = psd(recs)
        plt.figure(figsize=(6, 3.5)); plt.semilogy(f, p0, label="before"); plt.semilogy(f, p1, label=a.pipeline + " after")
        plt.xlabel("Hz"); plt.ylabel("PSD"); plt.legend(); plt.tight_layout()
        plt.savefig(a.out.replace(".npz", "_psd.png"), dpi=150)
    except Exception as e:
        print("psd plot skipped:", e)

    print(f"category={a.category} pipeline={a.pipeline} classes={len(classes)} {classes}")
    print(f"unreadable files skipped: {len(skipped)}  | subjects dropped (incomplete): {dropped}")
    print(f"kept subjects: {len(keep)}  recordings: {rec_id}  windows: {X.shape}  "
          f"({a.win} samples = {a.win/FS*1000:.0f} ms, stride {a.stride}) per-class: {np.bincount(ys).tolist()}")
    if a.pipeline == "fdf":
        print("ICA components removed per subject:", report["ica_components_removed_per_subject"])
    print("band power before -> after:", {k: (f"{v['before']:.3g}", f"{v['after']:.3g}")
                                          for k, v in report.get("band_power_before_after", {}).items()})
    print(f"saved {a.out} and {rp}")


if __name__ == "__main__":
    main()
