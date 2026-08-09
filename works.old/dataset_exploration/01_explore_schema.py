"""
Schema explorer. RAM small (~8GB), file huge (900MB) cos ~44 cols hold
long Bengali sentences instead of 0/1. Script no load full file -
just peek header + few rows, classify col type, write column_groups.csv.
"""
import csv

TRAIN = "dataset/train.csv"
TEST = "dataset/test.csv"
OUT = "dataset_exploration/column_groups.csv"

SAMPLE_ROWS = 300


def is_bool_text(values):
    # bool-text col: values long str start with Bengali yes/no word
    for v in values:
        if v is None or v == "":
            continue
        if len(v) > 15 and (v.startswith("হ্যাঁ") or v.startswith("না")):
            return True
    return False


def classify(name):
    if name == "Your_Target_Column":
        return "target"
    if name.startswith("base_"):
        return "base_attribute"
    if name.startswith("cost_"):
        return "financial"
    if name.startswith("trend_"):
        return "trend_pct"
    if name.startswith("sensor_"):
        return "sensor_reading"
    if name.startswith("count_"):
        return "operational_count"
    if name.startswith(("has_", "is_", "lacks_", "are_")):
        return "boolean_text_flag"
    if name.startswith(("num_", "num_var", "num_op", "num_aport", "num_compra",
                         "num_ent", "num_med", "num_reemb", "num_sal",
                         "num_trasp", "num_venta")):
        return "obfuscated_numeric"
    return "other"


def main():
    with open(TRAIN, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        header = next(r)
        sample = [next(r) for _ in range(SAMPLE_ROWS)]

    cols_by_idx = {i: [row[i] for row in sample] for i in range(len(header))}

    with open(TEST, encoding="utf-8-sig") as f:
        test_header = next(csv.reader(f))
    test_set = set(test_header)

    rows_out = []
    for i, name in enumerate(header):
        group = classify(name)
        vals = cols_by_idx[i]
        looks_bool_text = is_bool_text(vals)
        if looks_bool_text:
            group = "boolean_text_flag"
        uniq = set(vals)
        in_test = name in test_set
        example = next((v for v in vals if v), "")
        example = (example[:80] + "...") if len(example) > 80 else example
        rows_out.append({
            "idx": i,
            "column": name,
            "group": group,
            "n_unique_in_sample": len(uniq),
            "in_test": in_test,
            "example_value": example,
        })

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    # group counts summary
    from collections import Counter
    cnt = Counter(r["group"] for r in rows_out)
    with open("dataset_exploration/schema_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Total columns: {len(header)}\n")
        f.write(f"Train rows (excl header): see wc -l\n")
        f.write("Group counts:\n")
        for g, c in cnt.most_common():
            f.write(f"  {g}: {c}\n")
        f.write("\nBoolean-text flag columns (Bengali yes/no sentences instead of 0/1):\n")
        for r_ in rows_out:
            if r_["group"] == "boolean_text_flag":
                f.write(f"  - {r_['column']}\n")

    print("wrote", OUT)
    print("wrote dataset_exploration/schema_summary.txt")


if __name__ == "__main__":
    main()
