import json
import shutil

src_nb = 'candidates/synthetic/synthetic-fixissuesv2-version-3.ipynb'
dst_nb = 'final-submission/FINAL_SUBMISSION.ipynb'

shutil.copyfile(src_nb, dst_nb)

with open(dst_nb, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = cell['source']
        for i, line in enumerate(source):
            if 'DUO_LEVELING' in line:
                source[i] = line.replace('DUO_LEVELING', 'NawrizTurjo')
    elif cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "'probe_thresholds':" in line:
                source[i] = "    'probe_thresholds': [0.325, 0.35, 0.36, 0.375, 0.39, 0.40, 0.42, 0.45, 0.475, 0.495],\n"

with open(dst_nb, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Restored and updated FINAL_SUBMISSION.ipynb successfully!")
