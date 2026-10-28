# 🧠 Imagined-Speech EEG Classification
### EEGNet vs CNN–BiLSTM | Kumar Dataset

This repository contains experiments on imagined-character EEG signals from the Kumar dataset. It compares a compact **EEGNet baseline** with a **CNN–BiLSTM model** using random-window benchmarking and leave-one-subject-out (LOSO) evaluation.

> [!IMPORTANT]
> Calibration experiments are reported separately and must not be interpreted as evidence of subject-independent decoding.

---

## 📊 Main Findings

The current paper reports the following results for the ten-class character subset.

| Evaluation Protocol | Preprocessing | Model | Accuracy | Macro-F1 |
|:---|:---|:---|---:|---:|
| Random-window split | ICA-assisted 4–15 Hz frequency-domain band-reject (FDF) | EEGNet | 16.14% | 14.73% |
| Random-window split | ICA-assisted 4–15 Hz frequency-domain band-reject (FDF) | **CNN–BiLSTM** | **52.72%** | **52.77%** |
| LOSO (23 subjects) | FDF | EEGNet | 10.14 ± 2.48% | 6.15 ± 2.30% |
| LOSO (23 subjects) | FDF | CNN–BiLSTM | 10.06 ± 3.03% | 6.46 ± 2.15% |
| LOSO (23 subjects) | 0.5–40 Hz band-pass | EEGNet | 9.68 ± 2.05% | 5.51 ± 1.99% |
| LOSO (23 subjects) | 0.5–40 Hz band-pass | CNN–BiLSTM | 9.69 ± 2.30% | 5.26 ± 1.66% |

### Interpreting the results

- **Chance-level accuracy:** 10% for ten classes.
- **Random-window benchmark:** Exploratory only, not evidence of unseen-subject generalisation.
- **LOSO evaluation:** Both models remain near chance.
- **Windowing:** 32 samples (250 ms) per window, stride 16 samples (125 ms), with 50% overlap.

> [!WARNING]
> Overlapping windows from the same recording can appear in different random-split partitions. The random-window comparison uses one split/seed and is vulnerable to data leakage. Its results should not be presented as subject-independent or recording-independent performance.

The FDF pipeline fits ICA per subject using that subject's unlabelled recordings, including the held-out subject's recordings. No held-out labels are used, but this constitutes **transductive signal preprocessing** and should be disclosed when describing the protocol.

---

## 🗂️ Dataset

The archive contains three categories, each with ten classes:

| Category | Subjects in the archive copy |
|:---|---:|
| Char | 23 |
| Digit | 21 |
| Image | 21 |

Each subject has one approximately 12-second EDF recording per class, recorded using 14 EEG channels at 128 Hz.

In the archive copy used for these experiments, 34 files were corrupted: Digit and Image recordings for subjects 0 and 10.

> [!NOTE]
> The dataset is not included in this repository. Ensure you have permission to access and process it, follow the applicable dataset terms, and do not commit raw EDF files or private dataset copies to GitHub.

---

## 📁 Repository Structure

```text
imagined-speech-eeg-classification/
├── src/
│   ├── models.py
│   ├── data.py
│   ├── signal_proc.py
│   ├── protocols.py
│   ├── train.py
│   ├── calibrate.py
│   └── diagnose.py
├── scripts/
│   ├── prepare_kumar.py
│   ├── run_random_split_comparison.py
│   ├── fill_random_results.py
│   └── report.py
├── requirements.txt
└── README.md
```

| File | Purpose |
|:---|:---|
| `models.py` | EEGNet, EEGNetAttention, CNNBiLSTM, and shape tracing |
| `data.py` | NPZ loading and normalisation helpers |
| `signal_proc.py` | ICA artefact removal, FFT band-reject, and band-pass filters |
| `protocols.py` | LOSO/calibration splits and recording-level accuracy helpers |
| `train.py` | LOSO training and evaluation |
| `calibrate.py` | LOSO plus subject-calibration experiments |
| `diagnose.py` | Pre-flight diagnostics |
| `prepare_kumar.py` | EDF archive to NPZ preprocessing |
| `run_random_split_comparison.py` | Matched random-window model comparison |
| `fill_random_results.py` | Populate the LaTeX results table from JSON |
| `report.py` | Tables and plots from saved results |

---

## ⚙️ Setup

Use Python 3.10 or newer where possible.

### 1. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the following commands from the repository root. Replace example paths with the actual location of your dataset archive. `/content/archive.zip` is an example path for Google Colab.

---

## 🚀 Experimental Workflow

### 1. Prepare the data

**Basic 0.5–40 Hz band-pass, 1-second windows**

```bash
python scripts/prepare_kumar.py --raw /content/archive.zip --category Char --pipeline basic --win 128 --stride 64 --out data/char_basic_w128.npz
```

**ICA + 4–15 Hz FFT band-reject, 250 ms windows**

```bash
python scripts/prepare_kumar.py --raw /content/archive.zip --category Char --pipeline fdf --win 32 --stride 16 --out data/char_fdf_w32.npz
```

The preparation script writes an NPZ dataset and companion report/PSD files. Keep generated data and large outputs out of version control unless you have a specific, permitted reason to publish them.

To prepare other categories, replace `Char` with `Digit` or `Image` and choose a corresponding output filename.

### 2. Run diagnostics

Run the pre-flight diagnostic before a long training job:

```bash
python -m src.diagnose --data data/char_fdf_w32.npz --model cnn_bilstm --norm train --sched plateau --monitor loss --clip 1.0
```

This checks the dataset and training setup; it does not replace evaluation on real held-out subjects.

### 3. Run LOSO evaluation

**EEGNet with basic preprocessing**

```bash
python -m src.train --data data/char_basic_w128.npz --model eegnet --norm trial --seed 1
```

**CNN–BiLSTM with FDF preprocessing**

```bash
python -m src.train --data data/char_fdf_w32.npz --model cnn_bilstm --norm train --sched plateau --monitor loss --clip 1.0 --batch_size 256 --epochs 40 --patience 8 --seed 1
```

These commands illustrate experimental configurations. For exact reproduction, use the full settings and preprocessing variant documented in the paper and ensure the NPZ files were generated from the intended data and code version.

Results are aggregated over held-out subjects. Standard deviations across subjects are not standard deviations across random seeds.

### 4. Run LOSO with calibration

Calibration is a separate experiment.

**Temporal calibration split**

```bash
python -m src.calibrate --data data/char_fdf_w32.npz --model cnn_bilstm --norm train --sched plateau --monitor loss --clip 1.0 --batch_size 256 --epochs 40 --patience 8 --cal_split temporal
```

The script also supports `--cal_split random`. Use it only as a clearly labelled comparison.

> [!CAUTION]
> Calibration and test windows are taken from the same recordings, and each subject has only one recording per class. Even when a temporal split avoids shared samples, recording identity and session-specific properties can remain informative. Calibration scores are therefore diagnostic or upper-bound results, not proof of robust imagined-character decoding.

### 5. Run the matched random-window comparison

```bash
python scripts/run_random_split_comparison.py --data data/char_fdf_w32.npz --out results_random --seed 1
```

The script creates a stratified 70% training / 10% validation / 20% test split and reuses the same split indices for both models.

Default settings:

| Parameter | Value |
|:---|:---|
| Optimiser | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Batch size | 128 |
| Maximum epochs | 30 |
| Early-stopping patience | 8 |
| Dropout | 0.3 |
| Gradient clipping | 1.0 |
| Seed | 1 |

Normalisation statistics are fitted on the training partition when `--norm train` is used.

> [!WARNING]
> This experiment is intentionally leakage-prone because overlapping windows from the same source recording can occur across training, validation, and test partitions. Treat it as a matched within-dataset comparison only.

Do not use synthetic smoke-test outputs as scientific results. The script's `--synthetic` option is for checking that the pipeline runs, not for populating a paper.

---

## 📈 Outputs and Reporting

Depending on the script and options, outputs include:

- Per-subject CSV files and summary JSON
- Confusion matrices and training histories
- Comparison tables and split/settings metadata
- Recording-level accuracy
- Parameter counts and calibration results

For the random-window results, the helper can populate a LaTeX table from the JSON produced by the comparison script:

```bash
python scripts/fill_random_results.py --help
```

### Reporting checklist

- [ ] Accuracy and macro precision, recall, and F1.
- [ ] Mean ± standard deviation across held-out subjects for LOSO.
- [ ] Per-subject scores and confusion matrix, when available.
- [ ] Recording-level accuracy, clearly labelled and interpreted cautiously.
- [ ] Training/validation loss and accuracy, parameter count, split method, seed, preprocessing, window length, and stride.
- [ ] Calibration split type and results of any appropriate permuted-label control.

For ten-class classification, chance accuracy is 10% and uniform-prediction cross-entropy is approximately `ln(10) = 2.303`.

Preserve the exact settings and output files supporting every reported number. Do not report per-subject results or confusion matrices unless they were generated from the actual experiment.

---

## ⚠️ Limitations

- Random-window evaluation is vulnerable to leakage from overlapping windows and recording-specific signals.
- The random-window model comparison uses one split/seed; seed-to-seed variability has not been established.
- LOSO performance is near chance in the reported experiments.
- EEGNet's 64-sample temporal kernel is longer than the 32-sample input windows in the FDF experiment; the architecture was not adapted to those short windows.
- The FDF implementation uses a heuristic ICA-component rejection criterion and approximates, rather than exactly reproduces, the cited preprocessing method.
- In FDF LOSO, ICA is fitted using the held-out subject's unlabelled recordings. This is transductive signal preprocessing, although held-out labels are not used.
- There is one recording per subject and class, limiting the ability to separate class information from recording/session identity.
- Only the ten-class character subset is used for the principal reported comparison.

---

## 📚 Citation and Responsible Use

If you use this code or reproduce these experiments, cite the relevant dataset and methods papers listed in the accompanying research manuscript.

Clearly distinguish the random-window benchmark, LOSO evaluation, and calibration experiments. Do not describe these results as state-of-the-art or as evidence of reliable real-world imagined-speech decoding.
