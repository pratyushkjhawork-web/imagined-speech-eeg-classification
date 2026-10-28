"""Fill the result placeholders in the random-window paper from a real run JSON.

Example:
python scripts/fill_random_results.py \
  --results results_random/char_fdf_w32_random_split_comparison_seed1.json \
  --paper ../imagined_speech_random_window_comparison_pending.tex \
  --output ../imagined_speech_random_window_comparison_results.tex
"""
import argparse, json, re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--results', required=True, help='comparison JSON produced by run_random_split_comparison.py')
p.add_argument('--paper', required=True, help='paper .tex containing the placeholder table')
p.add_argument('--output', required=True, help='path for the filled .tex file')
a = p.parse_args()
data = json.loads(Path(a.results).read_text())
models = {row['model']: row for row in data['models']}
text = Path(a.paper).read_text()

def pct(key, row):
    v = row.get(key)
    return '--' if v is None else f'{100*float(v):.2f}'

def replace_row(source_text, model_label, key):
    row = models[key]
    # The values in the table are percentages. The final column is recording-level accuracy.
    new = (f"{model_label} & {int(row['params']):,} & {pct('accuracy', row)} & "
           f"{pct('precision', row)} & {pct('recall', row)} & {pct('f1', row)} & "
           f"{pct('recording_level_accuracy', row)} \\\\")
    pattern = rf'^{re.escape(model_label)}\s*&.*?\\\\$'
    out, n = re.subn(pattern, lambda _: new, source_text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f'Could not locate exactly one table row for {model_label!r}; found {n}')
    return out

text = replace_row(text, 'EEGNet', 'eegnet')
text = replace_row(text, 'CNN--BiLSTM', 'cnn_bilstm')
text = text.replace('All red entries are placeholders and must be replaced from \\texttt{results\\_random/*\\_random\\_split\\_comparison\\_seed1.json} after the real-data run. They are not measured results.', 'Values are computed from the matched random-window run; they describe this split only and are not subject-independent estimates.')
text = text.replace('The final metrics must be populated from the real-data run before submission.', 'The reported metrics describe the matched random-window split and must be interpreted with the leakage limitations discussed above.')
Path(a.output).write_text(text)
print(f'Wrote {a.output}')
