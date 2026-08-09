"""
Stream train/test csv once. Decode Bengali yes/no bool-text cols to 0/1,
keep numeric cols as float. Discard raw text after decode -> output tiny
compact csv (converted_train.csv / converted_test.csv) for modeling.
Also count how many rows had unexpected (non yes/no) bool values.
"""
import csv

TRAIN = "dataset/train.csv"
TEST = "dataset/test.csv"
SCHEMA = "dataset_exploration/column_groups.csv"
TARGET = "Your_Target_Column"

YES = "হ্যাঁ"
NO = "না"


def load_bool_cols():
    bool_cols = set()
    with open(SCHEMA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["group"] == "boolean_text_flag":
                bool_cols.add(row["column"])
    return bool_cols


def decode_bool(v):
    if v.startswith(YES):
        return "1"
    if v.startswith(NO):
        return "0"
    return ""  # unexpected -> empty/missing


def convert(path, bool_cols, out_path):
    n_unexpected = 0
    n_rows = 0
    with open(path, encoding="utf-8-sig") as f, \
         open(out_path, "w", newline="", encoding="utf-8") as out:
        r = csv.reader(f)
        header = next(r)
        w = csv.writer(out)
        w.writerow(["id"] + header)
        for row_idx, row in enumerate(r):
            new_row = [str(row_idx)]
            for i, v in enumerate(row):
                if header[i] in bool_cols:
                    d = decode_bool(v)
                    if d == "":
                        n_unexpected += 1
                    new_row.append(d)
                else:
                    new_row.append(v)
            w.writerow(new_row)
            n_rows += 1
    return n_rows, n_unexpected


def main():
    bool_cols = load_bool_cols()
    report = []
    for name, path in [("train", TRAIN), ("test", TEST)]:
        out_path = f"dataset_exploration/converted_{name}.csv"
        n_rows, n_unexpected = convert(path, bool_cols, out_path)
        report.append(f"{name}: rows={n_rows}, unexpected_bool_values={n_unexpected}, out={out_path}")
        print(report[-1])

    import os
    with open("dataset_exploration/conversion_report.txt", "w", encoding="utf-8") as f:
        f.write("Bool-text columns decoded (হ্যাঁ->1, না->0):\n")
        for c in sorted(bool_cols):
            f.write(f"  - {c}\n")
        f.write("\n")
        for line in report:
            f.write(line + "\n")
        for name in ["train", "test"]:
            p = f"dataset_exploration/converted_{name}.csv"
            f.write(f"{p} size: {os.path.getsize(p)/1e6:.1f} MB\n")


if __name__ == "__main__":
    main()
