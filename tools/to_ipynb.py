"""Convert a `# %%` cell-marked Python file into a Jupyter notebook."""
import json
import pathlib
import sys

src_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
lines = src_path.read_text(encoding="utf-8").splitlines()

cells = []
cur_type, cur_lines = None, []


def flush():
    if cur_type is None:
        return
    body = cur_lines[:]
    while body and not body[-1].strip():
        body.pop()
    while body and not body[0].strip():
        body.pop(0)
    if not body:
        return
    if cur_type == "markdown":
        body = [ln[2:] if ln.startswith("# ") else (ln[1:] if ln == "#" else ln) for ln in body]
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": [l + "\n" for l in body[:-1]] + [body[-1]]})
    else:
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [],
                      "source": [l + "\n" for l in body[:-1]] + [body[-1]]})


for ln in lines:
    if ln.startswith("# %% [markdown]"):
        flush()
        cur_type, cur_lines = "markdown", []
    elif ln.startswith("# %%"):
        flush()
        cur_type, cur_lines = "code", []
    else:
        cur_lines.append(ln)
flush()

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0",
                          "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                          "file_extension": ".py"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
n_code = len(cells) - n_md
print(f"wrote {out_path}  ({len(cells)} cells: {n_md} markdown, {n_code} code)")
