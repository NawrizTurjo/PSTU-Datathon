import json

nb_path = 'final-submission/FINAL_SUBMISSION.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 2 CFG probe_thresholds if present
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "'probe_thresholds':" in line:
                source[i] = "    'probe_thresholds': [0.325, 0.35, 0.36, 0.375, 0.39, 0.40, 0.42, 0.45, 0.475, 0.495],\n"

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated FINAL_SUBMISSION.ipynb successfully!")
