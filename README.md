Imagined-Speech EEG Classification: EEGNet vs CNN–BiLSTM
This repository contains code for experiments on imagined-character EEG from the Kumar dataset. It compares a compact EEGNet baseline with a CNN–BiLSTM model under a random-window benchmark and leave-one-subject-out (LOSO) evaluation. Calibration experiments are separate and must not be interpreted as evidence of subject-independent decoding.
Main findings
The current paper reports the following results for the ten-class character subset:
Evaluation protocol	Preprocessing	Model	Accuracy	Macro-F1
Stratified random-window split	ICA-assisted 4–15 Hz frequency-domain band-reject (FDF)	EEGNet	16.14%	14.73%
Stratified random-window split	ICA-assisted 4–15 Hz frequency-domain band-reject (FDF)	CNN–BiLSTM	52.72%	52.77%
LOSO, mean ± SD across 23 held-out subjects	FDF	EEGNet	10.14 ± 2.48%	6.15 ± 2.30%
LOSO, mean ± SD across 23 held-out subjects	FDF	CNN–BiLSTM	10.06 ± 3.03%	6.46 ± 2.15%
LOSO, mean ± SD across 23 held-out subjects	0.5–40 Hz band-pass	EEGNet	9.68 ± 2.05%	5.51 ± 1.99%
LOSO, mean ± SD across 23 held-out subjects	0.5–40 Hz band-pass	CNN–BiLSTM	9.69 ± 2.30%	5.26 ± 1.66%


There are ten classes, so chance-level accuracy is 10%. The random-window numbers are exploratory only and are not evidence of unseen-subject generalisation. Windows are 32 samples (250 ms) long with a stride of 16 samples (125 ms), meaning adjacent windows share 50% of their samples. Windows from the same recording can appear in different partitions. The random-window comparison uses one split/seed.

LOSO scores remain near chance for both models. The FDF preprocessing fits ICA per subject using that subject's unlabelled recordings, including the held-out subject's recordings. No held-out labels are used, but this is transductive signal preprocessing and should be disclosed when describing the protocol. See the paper for the full methodology, limitations, and discussion.
Dataset
The archive contains three categories (Char, Digit, Image), each with ten classes. Each subject has one approximately 12-second EDF recording per class, recorded using 14 EEG channels at 128 Hz. In the archive copy used for these experiments, 34 files were corrupted: Digit and Image recordings for subjects 0 and 10. The resulting counts are 23 subjects for Char and 21 subjects each for Digit and Image.

The dataset itself is not included in this repository. Ensure that you have permission to access and process it, and follow the dataset's applicable terms. Do not commit raw EDF files or private dataset copies to GitHub.
Repository layout
src/
  models.py        EEGNet, EEGNetAttention, CNNBiLSTM, and shape tracing
  data.py          NPZ loading and normalisation helpers
  signal_proc.py   ICA artefact removal, FFT band-reject, and band-pass filters
  protocols.py     LOSO/calibration splits and recording-level accuracy helpers
  train.py         LOSO training and evaluation
  calibrate.py     LOSO plus subject-calibration experiments
  diagnose.py      Pre-flight diagnostics
scripts/
  prepare_kumar.py                 EDF archive to NPZ preprocessing
  run_random_split_comparison.py   Matched random-window model comparison
  fill_random_results.py           Populate the LaTeX results table from JSON
  report.py                        Tables and plots from saved results
requirements.txt
README.md
Setup
Use Python 3.10 or newer where possible. From the repository root, create and activate a virtual environment, then install dependencies:
python -m venv .venv
Windows PowerShell:
.venv\Scripts\Activate.ps1
macOS/Linux:
source .venv/bin/activate
Install the listed dependencies:
python -m pip install --upgrade pip
pip install -r requirements.txt
Commands below should be run from the repository root. Replace example paths with the actual location of your dataset archive. The /content/archive.zip path is an example for a Google Colab environment.
1. Prepare the data
Prepare the character subset using the basic 0.5–40 Hz band-pass pipeline and 1-second windows:
python scripts/prepare_kumar.py --raw /content/archive.zip --category Char --pipeline basic --win 128 --stride 64 --out data/char_basic_w128.npz
Prepare the character subset using ICA plus a 4–15 Hz FFT band-reject pipeline and 250 ms windows:
python scripts/prepare_kumar.py --raw /content/archive.zip --category Char --pipeline fdf --win 32 --stride 16 --out data/char_fdf_w32.npz
The preparation script writes an NPZ dataset and companion report/PSD files. Keep generated data and large outputs out of version control unless you have a specific, permitted reason to publish them.

To prepare the other categories, replace Char with Digit or Image and choose a corresponding output filename.
2. Run diagnostics
Run the pre-flight diagnostic before a long training job:
python -m src.diagnose --data data/char_fdf_w32.npz --model cnn_bilstm --norm train --sched plateau --monitor loss --clip 1.0
This helps check the dataset and training setup; it does not replace evaluation on the real held-out subjects.
3. Run LOSO evaluation
Example EEGNet run using the basic preprocessing dataset:
python -m src.train --data data/char_basic_w128.npz --model eegnet --norm trial --seed 1
Example CNN–BiLSTM run using the FDF dataset:
python -m src.train --data data/char_fdf_w32.npz --model cnn_bilstm --norm train --sched plateau --monitor loss --clip 1.0 --batch_size 256 --epochs 40 --patience 8 --seed 1
These commands illustrate configurations used in experiments; for an exact reproduction, use the full settings and preprocessing variant documented in the paper and ensure that the NPZ files were generated from the intended data and code version. Results are aggregated over held-out subjects. Standard deviations across subjects are not standard deviations across random seeds.
4. Run LOSO with calibration (separate experiment)
Temporal calibration split:
python -m src.calibrate --data data/char_fdf_w32.npz --model cnn_bilstm --norm train --sched plateau --monitor loss --clip 1.0 --batch_size 256 --epochs 40 --patience 8 --cal_split temporal
The calibration script also supports --cal_split random. Use it only as a clearly labelled comparison with the limitations stated below.

Calibration caveat: calibration and test windows are taken from the same recordings, and each subject has only one recording per class. Even when a temporal split avoids shared samples, recording identity and session-specific properties can remain informative. Calibration scores should therefore be treated as an upper-bound or diagnostic result, not proof that imagined characters are being decoded robustly.
5. Run the matched random-window comparison
From the repository root:
python scripts/run_random_split_comparison.py --data data/char_fdf_w32.npz --out results_random --seed 1
The script creates a stratified 70% training / 10% validation / 20% test split and reuses the same split indices for both models. The default settings are AdamW, learning rate 0.001, weight decay 0.01, batch size 128, up to 30 epochs, patience 8, dropout 0.3, gradient clipping 1.0, and seed 1. Normalisation statistics are fitted on the training partition when --norm train is used.

Important: this experiment is intentionally leakage-prone because overlapping windows from the same source recording can occur in the training, validation, and test partitions. It is useful as a matched within-dataset comparison only. Do not present its accuracy as subject-independent or recording-independent performance.

Do not use synthetic smoke-test outputs as scientific results. The script's --synthetic option is for checking that the pipeline runs, not for populating a paper.
Outputs
Depending on the script and options, outputs include per-subject CSV files, summary JSON, confusion matrices, training histories, comparison tables, and split/settings metadata. Preserve the exact settings and output files needed to support any reported number. Do not report per-subject results or confusion matrices unless they were generated from the actual experiment.

For the random-window results, the helper can populate a LaTeX table from the JSON produced by the comparison script. Check the helper's --help output and provide the correct paths for your local paper files, for example:
python scripts/fill_random_results.py --help
Reporting checklist
For each experiment, record:
- Accuracy and macro precision, recall, and F1 (mean ± standard deviation across held-out subjects for LOSO).
- Per-subject scores and confusion matrix when available.
- Recording-level accuracy, clearly labelled and interpreted cautiously.
- Training/validation loss and accuracy, parameter count, split method, seed, preprocessing, window length, and stride.
- For calibration, the calibration split type and the results of any appropriate permuted-label control.
For ten-class classification, chance accuracy is 10% and uniform-prediction cross-entropy is approximately ln(10) = 2.303.
Limitations
- The random-window protocol is vulnerable to leakage from overlapping windows and recording-specific signals.
- The random-window model comparison uses one split/seed; its seed-to-seed variability has not been established.
- LOSO performance is near chance in the reported experiments.
- EEGNet's 64-sample temporal kernel is longer than the 32-sample input windows in the FDF experiment; the architecture was not adapted to those short windows.
- The FDF implementation uses a heuristic ICA-component rejection criterion and approximates, rather than exactly reproduces, the cited preprocessing method.
- In FDF LOSO, ICA is fitted using the held-out subject's unlabelled recordings. This is transductive signal preprocessing, although the held-out labels are not used.
- There is one recording per subject and class, limiting the ability to separate class information from recording/session identity.
- Only the ten-class character subset is used for the principal reported comparison.
Citation and responsible use
If you use this code or reproduce these experiments, cite the relevant dataset and methods papers listed in the accompanying research manuscript. Clearly distinguish the random-window benchmark, LOSO evaluation, and calibration experiments. Do not describe the results as state-of-the-art or as evidence of reliable real-world imagined-speech decod