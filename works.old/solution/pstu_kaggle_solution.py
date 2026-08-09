# %% [markdown]
# # PSTU Datathon — Predictive Maintenance Solution
#
# Predicting 7-day critical failure in off-grid solar water stations.
#
# **This notebook is built from measured EDA findings, not assumptions.** Key facts it relies on:
#
# | Finding | Value | Consequence |
# |---|---|---|
# | Boolean columns stored as Bengali sentences | 63 columns | Streaming decoder below (910 MB → in-memory numeric) |
# | Hidden sentinel | `-999999` in `base_number_of_dependent_farmers` | → `NaN`, handled natively by GBDTs |
# | Zero-information columns | 6 constant + 6 duplicate | dropped |
# | Train/test shift | adversarial AUC 0.4985 | plain `StratifiedKFold` is safe |
# | Station id | none recoverable | **not** `GroupKFold` |
# | Metric collapses to | `0.30·F1 + 0.25·AUC + 0.15·P + 0.20·R + 0.10·S` | recall outweighs precision |
# | All-ones baseline | **0.4388** | any model must beat this |
# | Threshold tuning | **+0.018** composite | never submit a 0.5 cutoff |
#
# Baselines to beat: RandomForest `0.5167`, HistGradientBoosting `0.5269`.
#
# Runtime: roughly **15–30 minutes on CPU**. Do not enable a GPU — at 48k × 286 it is slower
# than CPU for gradient-boosted trees.

# %%
import os
import gc
import csv
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------
# CONFIG — everything tunable lives here
# ----------------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
N_SEEDS = 2            # seed-averaging per model. 3+ is better if you have the time.

USE_LGB = True
USE_XGB = True
USE_CAT = True

# Feature-engineering blocks. Toggle these to A/B test their contribution:
# run once with all False to get a raw-feature score, then enable one at a time.
FE_ROW_AGGREGATES = True    # Santander row-stats — highest prior, try first
FE_GROUP_AGGREGATES = True  # same, per semantic column group
FE_NET_FLAGS = True         # has_X - lacks_X consolidation
FE_DOMAIN_RATIOS = True     # physical / financial ratios

# Class weighting. Measured note: because AUC is rank-based and the threshold is tuned
# afterwards, weighting usually matters less than expected here. Worth an A/B.
USE_CLASS_WEIGHT = False

TARGET = "Your_Target_Column"
OUTPUT_PATH = "submission.csv"

np.random.seed(SEED)
print("config loaded")

# %% [markdown]
# ## 1. Locate the data
#
# Kaggle mounts competition data read-only. The path is auto-detected so this notebook
# also runs locally without edits.

# %%
CANDIDATE_DIRS = [
    "/kaggle/input/competitions/pstu-data-craft-transforming-raw-data-into-impact",
    "/kaggle/input/pstu-data-craft-transforming-raw-data-into-impact",
    "/kaggle/input/pstu-datathon",
    "./dataset",
    "../dataset",
]

DATA_DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(os.path.join(d, "test.csv")):
        DATA_DIR = d
        break

if DATA_DIR is None:
    # last resort: walk /kaggle/input looking for train.csv
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "train.csv" in files and "test.csv" in files:
            DATA_DIR = root
            break

assert DATA_DIR is not None, (
    "Could not find train.csv/test.csv. Check the dataset is attached, then set DATA_DIR manually."
)

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
print(f"DATA_DIR = {DATA_DIR}")
print(f"  train: {os.path.getsize(TRAIN_CSV)/1e6:8.1f} MB")
print(f"  test : {os.path.getsize(TEST_CSV)/1e6:8.1f} MB")

# %% [markdown]
# ## 2. Streaming loader with Bengali boolean decode
#
# The raw `train.csv` is ~910 MB for only 48k rows because 63 boolean columns are stored as
# full Bengali sentences (`"হ্যাঁ, এই স্টেশনটিতে..."` = yes, `"না, ..."` = no) instead of `0`/`1`.
#
# Reading that straight into pandas creates 63 object columns of long strings and wastes
# several GB. Instead we stream the file once with `csv.reader`, decode each flag to `1`/`0`
# on the fly, and never materialise the text. Takes ~20–40 seconds and yields a ~55 MB
# float array.

# %%
YES, NO = "হ্যাঁ", "না"


def detect_bool_text_columns(path, sample_rows=500):
    """Find columns whose values are long Bengali yes/no sentences."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        samples = [[] for _ in header]
        for _, row in zip(range(sample_rows), reader):
            for i, v in enumerate(row):
                samples[i].append(v)

    bool_idx = set()
    for i, vals in enumerate(samples):
        for v in vals:
            if len(v) > 15 and (v.startswith(YES) or v.startswith(NO)):
                bool_idx.add(i)
                break
    return header, bool_idx


def _to_float(v):
    if v == "" or v == "NA" or v == "NaN":
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def load_decoded(path, bool_names):
    """Stream a CSV, decoding Bengali boolean columns to 1.0/0.0. Returns a float DataFrame."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        bool_idx = {i for i, name in enumerate(header) if name in bool_names}

        rows = []
        for row in reader:
            out = [0.0] * len(row)
            for i, v in enumerate(row):
                if i in bool_idx:
                    out[i] = 1.0 if v.startswith(YES) else (0.0 if v.startswith(NO) else np.nan)
                else:
                    out[i] = _to_float(v)
            rows.append(out)

    arr = np.asarray(rows, dtype=np.float64)
    return pd.DataFrame(arr, columns=header)


t0 = time.time()
header, bool_idx = detect_bool_text_columns(TRAIN_CSV)
BOOL_COLS = {header[i] for i in bool_idx}
print(f"detected {len(BOOL_COLS)} Bengali boolean-text columns")

train = load_decoded(TRAIN_CSV, BOOL_COLS)
test = load_decoded(TEST_CSV, BOOL_COLS)
print(f"loaded in {time.time()-t0:.1f}s   train={train.shape}  test={test.shape}")
print(f"memory: train {train.memory_usage(deep=True).sum()/1e6:.0f} MB, "
      f"test {test.memory_usage(deep=True).sum()/1e6:.0f} MB")

y = train[TARGET].astype(int).values
train = train.drop(columns=[TARGET])
print(f"positive rate: {y.mean():.4%}  ({y.sum()} of {len(y)})")

assert list(train.columns) == list(test.columns), "train/test column mismatch after load"
gc.collect()

# %% [markdown]
# ## 3. Preprocessing
#
# Three steps, each justified by a measured finding:
# 1. The `-999999` sentinel → `NaN` (a negative farmer count is physically impossible).
# 2. Drop 6 columns that are constant in **both** train and test.
# 3. Drop 6 columns that exactly duplicate another column.
#
# Missing values are deliberately **left as `NaN`** — LightGBM, XGBoost and CatBoost all
# handle them natively and learn a dedicated split direction.

# %%
SENTINEL_COL = "base_number_of_dependent_farmers"
SENTINEL_VALUE = -999999

CONSTANT_COLS = [
    "has_no_medium_term_fund_balance", "has_no_reimbursement_delta",
    "has_zero_grid_power_balance", "has_zero_medium_term_avg_balance",
    "has_zero_solar_efficiency_balance", "has_zero_water_tank_balance",
]

# keep the first of each identical pair, drop the second
DUPLICATE_COLS = [
    "is_pump_draw_dry",                     # == has_dust_accumulation_on_panels
    "count_battery_failures",               # == count_pump_motor_faults
    "trend_maintenance_claim_count_1y3",    # == trend_maintenance_cost_increase_1y3
    "trend_repair_claim_count_1y3",         # == trend_repair_cost_increase_1y3
    "trend_expense_transaction_count_1y3",  # == trend_outgoing_expense_increase_1y3
    "trend_internal_in_count_1y3",          # == trend_internal_transfer_in_1y3
]


def preprocess(df):
    df = df.copy()
    n_sentinel = int((df[SENTINEL_COL] == SENTINEL_VALUE).sum())
    df[SENTINEL_COL] = df[SENTINEL_COL].replace(SENTINEL_VALUE, np.nan)
    df = df.drop(columns=CONSTANT_COLS + DUPLICATE_COLS, errors="ignore")
    return df, n_sentinel


train, n_tr = preprocess(train)
test, n_te = preprocess(test)
print(f"sentinel -> NaN: {n_tr} train rows, {n_te} test rows")
print(f"after dropping 12 zero-information columns: {train.shape[1]} features")

BOOL_COLS = {c for c in BOOL_COLS if c in train.columns}
NUM_COLS = [c for c in train.columns if c not in BOOL_COLS]
print(f"  {len(BOOL_COLS)} boolean, {len(NUM_COLS)} numeric")

# %% [markdown]
# ## 4. Feature engineering
#
# Four blocks, each independently toggleable in CONFIG so you can measure their
# contribution one at a time.
#
# **Block A (row aggregates) has the strongest prior.** This dataset's numeric skeleton is
# the Santander Customer Satisfaction feature set, where row-wise sparsity statistics —
# especially the count of zeros — were consistently among the most predictive engineered
# features. 143 of 223 numeric columns here are ≥90% zero, so the same logic applies.

# %%
def classify_column(name):
    if name.startswith("base_"):
        return "base"
    if name.startswith("cost_"):
        return "financial"
    if name.startswith("trend_"):
        return "trend"
    if name.startswith("sensor_"):
        return "sensor"
    if name.startswith("count_"):
        return "count"
    if name.startswith("num_"):
        return "obfuscated"
    return "other"


COL_GROUPS = {}
for c in NUM_COLS:
    COL_GROUPS.setdefault(classify_column(c), []).append(c)
print({k: len(v) for k, v in COL_GROUPS.items()})

# has_X / lacks_X pairs confirmed by EDA. Measured behaviour: never both 1, but
# frequently both 0 — so "neither" is a real third state worth encoding.
FLAG_PAIRS = [
    ("has_primary_solar_inverter", "lacks_primary_solar_inverter"),
    ("has_battery_backup_system", "lacks_battery_backup_system"),
    ("is_salinity_sensor_active", "is_salinity_sensor_inactive"),
    ("is_submersible_pump_operational", "is_submersible_pump_non_operational"),
    ("has_solar_panel_cleaning_schedule", "lacks_solar_panel_cleaning_schedule"),
    ("has_remote_monitoring_system", "lacks_remote_monitoring_system"),
    ("is_groundwater_level_stable", "is_groundwater_level_fluctuating"),
    ("is_pump_motor_cool", "is_pump_motor_overheating"),
    ("has_auto_voltage_regulator", "lacks_auto_voltage_regulator"),
    ("has_flood_submersion_history", "has_no_flood_submersion_history"),
    ("is_local_technician_available", "is_local_technician_unavailable"),
    ("has_alternative_water_source", "lacks_alternative_water_source"),
    ("has_emergency_short_term_fund", "lacks_emergency_short_term_fund"),
    ("has_long_term_govt_subsidy", "lacks_long_term_govt_subsidy"),
    ("has_solar_charge_controller", "lacks_solar_charge_controller"),
]

RISK_FLAGS = [
    "is_pump_motor_overheating", "is_pipe_corroded_by_salt",
    "has_dust_accumulation_on_panels", "has_flood_submersion_history",
    "is_local_technician_unavailable", "lacks_battery_backup_system",
    "lacks_auto_voltage_regulator", "has_constant_charge_controller_issue",
    "lacks_remote_monitoring_system", "is_community_untrained_for_maintenance",
]

EPS = 1e-6


def engineer(df):
    df = df.copy()
    num = [c for c in NUM_COLS if c in df.columns]

    # --- Block A: row-wise aggregates over all numeric columns ---
    if FE_ROW_AGGREGATES:
        sub = df[num]
        df["agg_n_zeros"] = (sub == 0).sum(axis=1)
        df["agg_n_nonzero"] = (sub != 0).sum(axis=1)
        df["agg_n_negative"] = (sub < 0).sum(axis=1)
        df["agg_sum"] = sub.sum(axis=1)
        df["agg_mean"] = sub.mean(axis=1)
        df["agg_std"] = sub.std(axis=1)
        df["agg_max"] = sub.max(axis=1)
        df["agg_skew"] = sub.skew(axis=1)

    # --- Block A2: same, per semantic group ---
    if FE_GROUP_AGGREGATES:
        for grp, cols in COL_GROUPS.items():
            cols = [c for c in cols if c in df.columns]
            if len(cols) < 3:
                continue
            g = df[cols]
            df[f"{grp}_n_nonzero"] = (g != 0).sum(axis=1)
            df[f"{grp}_sum"] = g.sum(axis=1)
            df[f"{grp}_mean"] = g.mean(axis=1)
            df[f"{grp}_max"] = g.max(axis=1)

    # --- Block B: boolean net-flags ---
    if FE_NET_FLAGS:
        for pos, neg in FLAG_PAIRS:
            if pos in df.columns and neg in df.columns:
                df[f"net_{pos}"] = df[pos] - df[neg]
                df[f"unk_{pos}"] = ((df[pos] == 0) & (df[neg] == 0)).astype(np.int8)
        present = [c for c in RISK_FLAGS if c in df.columns]
        if present:
            df["risk_flag_count"] = df[present].sum(axis=1)

    # --- Block C: domain ratios and stress indices ---
    if FE_DOMAIN_RATIOS:
        age = df["base_station_installation_age_years"] + EPS

        for c, out in [("count_dry_run_events", "dry_runs_per_year"),
                       ("count_voltage_surge_events", "surges_per_year"),
                       ("count_major_repairs_total", "major_repairs_per_year"),
                       ("count_maintenance_visits_total", "maint_visits_per_year"),
                       ("count_overheating_events", "overheating_per_year")]:
            if c in df.columns:
                df[out] = df[c] / age

        # maintenance debt: time since service relative to how often it is normally serviced
        for since, total, out in [
            ("count_months_since_panel_cleaning", "count_solar_panel_cleanings", "cleaning_debt"),
            ("count_months_since_last_maintenance", "count_maintenance_visits_total", "maint_debt"),
            ("count_months_since_major_repair", "count_major_repairs_total", "repair_debt"),
        ]:
            if since in df.columns and total in df.columns:
                df[out] = df[since] / (df[total] + EPS)

        # financial fragility
        if "cost_total_repair_bdt" in df.columns:
            df["repair_cost_per_farmer"] = (
                df["cost_total_repair_bdt"] / (df[SENTINEL_COL].abs() + EPS))
        funding_cols = [c for c in ["cost_govt_grant_bdt", "cost_community_contribution_bdt",
                                    "cost_ngo_funding_3m_bdt", "cost_subsidy_received_bdt"]
                        if c in df.columns]
        if funding_cols:
            df["total_funding"] = df[funding_cols].sum(axis=1)
            if "cost_total_maintenance_bdt" in df.columns:
                df["funding_deficit"] = df["cost_total_maintenance_bdt"] - df["total_funding"]
            if "cost_total_repair_bdt" in df.columns:
                df["grant_to_repair_ratio"] = (
                    df["total_funding"] / (df["cost_total_repair_bdt"] + EPS))

        # environmental stress interactions
        for c, out in [("sensor_water_salinity_ppm", "salinity_x_age"),
                       ("sensor_motor_vibration_level_mm_s", "vibration_x_age"),
                       ("sensor_panel_surface_dust_index", "dust_x_age")]:
            if c in df.columns:
                df[out] = df[c] * age

        # month-over-month deltas: direction of travel, not just level
        for base in ["water_tank", "temp", "demand", "humidity", "vibration",
                     "irradiance", "short_runtime", "long_runtime"]:
            last = f"sensor_avg_{base}_last_month"
            prev = f"sensor_avg_{base}_2_months_ago"
            if last in df.columns and prev in df.columns:
                df[f"delta_{base}"] = df[last] - df[prev]
                df[f"ratio_{base}"] = df[last] / (df[prev] + EPS)

    return df.replace([np.inf, -np.inf], np.nan)


t0 = time.time()
X = engineer(train)
X_test = engineer(test)
print(f"feature engineering: {time.time()-t0:.1f}s")
print(f"features: {train.shape[1]} -> {X.shape[1]}  (+{X.shape[1]-train.shape[1]})")

assert list(X.columns) == list(X_test.columns), "train/test feature mismatch"
FEATURES = list(X.columns)
del train, test
gc.collect()

# %% [markdown]
# ## 5. The competition metric
#
# $$\text{Score} = 0.30 F_1 + 0.25 \text{AUC} + 0.15 P + 0.15 R + 0.10 \text{BalAcc} + 0.05 S$$
#
# Substituting $\text{BalAcc}=(R+S)/2$ gives the exactly equivalent
# $0.30F_1 + 0.25\text{AUC} + 0.15P + \mathbf{0.20}R + 0.10S$ — **recall outweighs precision**,
# which the official form hides by splitting recall's weight across two terms.
#
# The threshold search below is **exhaustive and vectorised**: sorting predictions once and
# taking cumulative sums gives the confusion matrix at *every possible* cut-point in $O(n)$,
# so there is no grid resolution to trade off. AUC is threshold-independent and computed once.

# %%
def composite_score(y_true, y_pred_binary, y_proba):
    """Exact official metric. Returns (score, components)."""
    y_true = np.asarray(y_true)
    y_pred_binary = np.asarray(y_pred_binary)
    tp = int(((y_true == 1) & (y_pred_binary == 1)).sum())
    fp = int(((y_true == 0) & (y_pred_binary == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_binary == 0)).sum())
    tn = int(((y_true == 0) & (y_pred_binary == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    bal = (rec + spec) / 2
    auc = roc_auc_score(y_true, y_proba)

    score = 0.30 * f1 + 0.25 * auc + 0.15 * prec + 0.15 * rec + 0.10 * bal + 0.05 * spec
    return score, dict(f1=f1, auc=auc, precision=prec, recall=rec,
                       specificity=spec, balanced_accuracy=bal)


def cutoff_curve(y_true, proba):
    """Composite score at *every* cut-point, in O(n log n). Returns (scores_by_k, auc).

    scores_by_k[i] is the composite obtained by labelling the top (i+1) rows positive.
    Sorting once and taking cumulative sums gives the full confusion matrix at every k,
    so there is no grid resolution to trade off. AUC is threshold-free -> computed once.
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    P = int(y_true.sum())
    N = n - P
    auc = roc_auc_score(y_true, proba)

    order = np.argsort(-proba, kind="mergesort")
    y_sorted = y_true[order]

    k = np.arange(1, n + 1)
    tp = np.cumsum(y_sorted)
    fp = k - tp
    fn = P - tp
    tn = N - fp

    with np.errstate(divide="ignore", invalid="ignore"):
        prec = tp / k
        rec = tp / P
        spec = tn / N
        denom = prec + rec
        f1 = np.where(denom > 0, 2 * prec * rec / np.where(denom > 0, denom, 1), 0.0)
    bal = (rec + spec) / 2

    scores = 0.30 * f1 + 0.25 * auc + 0.15 * prec + 0.15 * rec + 0.10 * bal + 0.05 * spec
    return scores, auc


def optimise_cutoff(y_true, proba):
    """Exhaustive argmax over every cut-point. Returns (best_k, best_rate, best_score)."""
    scores, _ = cutoff_curve(y_true, proba)
    best_i = int(np.argmax(scores))
    return best_i + 1, (best_i + 1) / len(y_true), float(scores[best_i])


def report(name, y_true, proba):
    auc = roc_auc_score(y_true, proba)
    k, rate, score = optimise_cutoff(y_true, proba)
    pred = np.zeros(len(y_true), dtype=int)
    pred[np.argsort(-proba, kind="mergesort")[:k]] = 1
    _, m = composite_score(y_true, pred, proba)
    print(f"{name:28s} AUC={auc:.4f}  composite={score:.4f}  "
          f"(rate={rate:.3f}  F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f})")
    return dict(name=name, auc=auc, composite=score, rate=rate)


# sanity check the metric against the two known degenerate baselines
_p = np.full(len(y), 0.05)
_s0, _ = composite_score(y, np.zeros(len(y), int), _p)
_s1, _ = composite_score(y, np.ones(len(y), int), _p)
print(f"metric check -> all-zeros {_s0 - 0.25*0.5 + 0.25*0.811:.4f} (expect ~0.3028), "
      f"all-ones {_s1 - 0.25*0.5 + 0.25*0.811:.4f} (expect ~0.4388)")

# %% [markdown]
# ## 6. Models
#
# LightGBM, XGBoost and CatBoost, each seed-averaged and 5-fold cross-validated on a
# **shared fold split** so the out-of-fold predictions can be blended and compared fairly.
#
# Early stopping is on fold **AUC**, not the composite — the composite is threshold-dependent
# and noisy, so we optimise ranking during training and handle the cut-point afterwards.

# %%
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
FOLDS = list(cv.split(X, y))
scale_pos_weight = (len(y) - y.sum()) / y.sum()
print(f"scale_pos_weight would be {scale_pos_weight:.1f} (enabled: {USE_CLASS_WEIGHT})")


def run_cv(fit_predict, name, n_seeds=N_SEEDS):
    """Run seed-averaged K-fold CV. fit_predict(X_tr,y_tr,X_va,y_va,seed) -> (val_pred, test_pred)."""
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    t0 = time.time()

    for s in range(n_seeds):
        seed = SEED + s * 1000
        oof_s = np.zeros(len(y))
        for fold, (tr_i, va_i) in enumerate(FOLDS):
            vp, tp = fit_predict(X.iloc[tr_i], y[tr_i], X.iloc[va_i], y[va_i], seed)
            oof_s[va_i] = vp
            test_pred += tp / (N_FOLDS * n_seeds)
        oof += oof_s / n_seeds
        print(f"  {name} seed {seed}: OOF AUC {roc_auc_score(y, oof_s):.4f}")

    print(f"  {name} done in {time.time()-t0:.0f}s")
    return oof, test_pred


oof_preds, test_preds, results = {}, {}, []

# %%
if USE_LGB:
    import lightgbm as lgb
    print(f"lightgbm {lgb.__version__}")

    def fit_lgb(X_tr, y_tr, X_va, y_va, seed):
        params = dict(
            objective="binary", metric="auc", learning_rate=0.03,
            num_leaves=31, min_child_samples=40,
            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
            lambda_l1=0.1, lambda_l2=1.0,
            n_estimators=3000, random_state=seed, n_jobs=-1, verbosity=-1,
        )
        if USE_CLASS_WEIGHT:
            params["scale_pos_weight"] = scale_pos_weight
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_test)[:, 1]

    oof_preds["lgb"], test_preds["lgb"] = run_cv(fit_lgb, "lightgbm")
    results.append(report("lightgbm", y, oof_preds["lgb"]))

# %%
if USE_XGB:
    import xgboost as xgb
    print(f"xgboost {xgb.__version__}")

    def fit_xgb(X_tr, y_tr, X_va, y_va, seed):
        params = dict(
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            learning_rate=0.03, max_depth=5, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
            n_estimators=3000, random_state=seed, n_jobs=-1,
            early_stopping_rounds=200,
        )
        if USE_CLASS_WEIGHT:
            params["scale_pos_weight"] = scale_pos_weight
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_test)[:, 1]

    oof_preds["xgb"], test_preds["xgb"] = run_cv(fit_xgb, "xgboost")
    results.append(report("xgboost", y, oof_preds["xgb"]))

# %% [markdown]
# CatBoost's ordered boosting is specifically designed to resist overfitting to noisy
# labels — relevant here, since 3.3% of train rows sit in duplicate groups with
# **conflicting targets** (identical features, different outcome). It is the slowest of
# the three; drop `USE_CAT` if you are time-constrained.

# %%
if USE_CAT:
    from catboost import CatBoostClassifier
    import catboost
    print(f"catboost {catboost.__version__}")

    def fit_cat(X_tr, y_tr, X_va, y_va, seed):
        params = dict(
            loss_function="Logloss", eval_metric="AUC",
            learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
            iterations=3000, random_seed=seed,
            early_stopping_rounds=200, verbose=False, allow_writing_files=False,
        )
        if USE_CLASS_WEIGHT:
            params["scale_pos_weight"] = scale_pos_weight
        m = CatBoostClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_test)[:, 1]

    oof_preds["cat"], test_preds["cat"] = run_cv(fit_cat, "catboost")
    results.append(report("catboost", y, oof_preds["cat"]))

# %% [markdown]
# ## 7. Ensemble — rank averaging
#
# The `Target_Probability` column feeds **only** AUC, and AUC depends only on *ordering*.
# So we blend **ranks, not probabilities** — this makes the blend immune to the three models
# being calibrated on different scales.
#
# Blend weights are chosen to maximise OOF **AUC** (the only thing this column affects) via a
# random search over the simplex — deterministic, and with no optimiser failure modes.

# %%
from scipy.stats import rankdata

model_names = list(oof_preds.keys())
print(f"blending: {model_names}")


def rank_blend(preds, weights):
    total = sum(weights)
    return sum(w * rankdata(p) / len(p) for w, p in zip(weights, preds)) / total


if len(model_names) == 1:
    best_w = [1.0]
    print("single model, no blending needed")
else:
    oof_list = [oof_preds[m] for m in model_names]
    rng = np.random.default_rng(SEED)
    best_w, best_auc = None, -1.0

    # include equal weights and each single model as candidates
    candidates = [np.ones(len(model_names))]
    candidates += list(np.eye(len(model_names)))
    candidates += list(rng.dirichlet(np.ones(len(model_names)), size=2000))

    for w in candidates:
        a = roc_auc_score(y, rank_blend(oof_list, w))
        if a > best_auc:
            best_auc, best_w = a, np.asarray(w, dtype=float)

    best_w = best_w / best_w.sum()
    print("optimal weights: " + ", ".join(f"{m}={w:.3f}" for m, w in zip(model_names, best_w)))
    print(f"blend OOF AUC: {best_auc:.4f}")

oof_blend = rank_blend([oof_preds[m] for m in model_names], best_w)
test_blend = rank_blend([test_preds[m] for m in model_names], best_w)

print()
for r in results:
    print(f"  {r['name']:28s} composite {r['composite']:.4f}")
final = report("BLEND", y, oof_blend)

# %% [markdown]
# ## 8. Cut-point selection
#
# Measured: tuning this is worth **+0.018 composite** versus a naive 0.5 threshold, and the
# optimum moves with the model (0.60 for RandomForest, 0.53 for HistGradientBoosting) — so
# it must be re-derived here rather than hardcoded.
#
# We select a **positive rate** rather than a probability threshold. For a fixed score vector
# the two are equivalent, but a rate transfers cleanly to the test set regardless of
# calibration, and cannot land inside a probability cliff.
#
# Two robustness checks below: per-fold stability, and how flat the optimum is.

# %%
n = len(y)
scores_by_k, _ = cutoff_curve(y, oof_blend)
argmax_k = int(np.argmax(scores_by_k))
best_score = float(scores_by_k[argmax_k])

print(f"argmax cut-point:      rate={  (argmax_k+1)/n:.4f}  composite={best_score:.4f}")
naive = (oof_blend >= 0.5).astype(int)
print(f"naive 0.5 threshold:   composite={composite_score(y, naive, oof_blend)[0]:.4f}")

# Robustness: the argmax can chase OOF noise inside a flat region. Take the CENTRE of the
# near-optimal plateau instead — same score, but far less sensitive to the exact OOF sample.
TOL = 5e-4
plateau = np.flatnonzero(scores_by_k >= best_score - TOL)
robust_k = int(np.median(plateau)) + 1
FINAL_RATE = robust_k / n
print(f"plateau ({len(plateau)} cut-points within {TOL} of best) spans rates "
      f"{(plateau[0]+1)/n:.4f}-{(plateau[-1]+1)/n:.4f}")
print(f"plateau-centred choice: rate={FINAL_RATE:.4f}  "
      f"composite={float(scores_by_k[robust_k-1]):.4f}")

# per-fold stability — wide scatter means the optimum is noise-driven
fold_rates = [optimise_cutoff(y[va_i], oof_blend[va_i])[1] for _tr, va_i in FOLDS]
print(f"\nper-fold optimal rates: {[f'{r:.3f}' for r in fold_rates]}")
print(f"  median {np.median(fold_rates):.4f}, std {np.std(fold_rates):.4f}")

print("\nsensitivity around the chosen rate:")
order_oof = np.argsort(-oof_blend, kind="mergesort")
for delta in [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04]:
    rate = FINAL_RATE + delta
    if not 0 < rate < 1:
        continue
    pred = np.zeros(n, dtype=int)
    pred[order_oof[:int(round(rate * n))]] = 1
    s, _ = composite_score(y, pred, oof_blend)
    print(f"  rate {rate:.3f}: composite {s:.4f}" + ("   <-- chosen" if delta == 0.0 else ""))

print(f"\nFINAL_RATE = {FINAL_RATE:.4f}")

# %% [markdown]
# ## 9. Submission
#
# The two columns are scored independently, so they are built independently:
# - `Target_Probability` → the rank-blended scores (drives AUC only)
# - `Target_Binary` → the top `FINAL_RATE` fraction by that score
#
# Probabilities are clipped strictly inside `(0, 1)` — the grader rejects exact `0.0`/`1.0`,
# and an "Evaluation Error" scores nothing at all.
#
# Note the submitted probabilities are **rank-normalised**, not calibrated. That is deliberate
# and safe: this column is used only for ROC-AUC, which depends purely on ordering, and rank
# values are valid floats strictly inside `(0, 1)`.

# %%
n_test = len(test_blend)
k_test = int(round(FINAL_RATE * n_test))

order_test = np.argsort(-test_blend, kind="mergesort")
binary = np.zeros(n_test, dtype=int)
binary[order_test[:k_test]] = 1

proba = np.clip(test_blend, 1e-6, 1 - 1e-6)

submission = pd.DataFrame({
    "id": np.arange(n_test, dtype=int),
    "Target_Binary": binary,
    "Target_Probability": proba,
})


def validate_submission(sub, expected_rows):
    assert list(sub.columns) == ["id", "Target_Binary", "Target_Probability"], "bad columns"
    assert len(sub) == expected_rows, f"expected {expected_rows} rows, got {len(sub)}"
    assert sub["id"].tolist() == list(range(expected_rows)), "id must be 0..n-1 in row order"
    assert sub["Target_Binary"].isin([0, 1]).all(), "binary must be 0/1"
    assert sub["Target_Probability"].between(0, 1, inclusive="neither").all(), \
        "probability must be strictly inside (0,1)"
    assert np.isfinite(sub["Target_Probability"]).all(), "probability has NaN/inf"
    return True


validate_submission(submission, n_test)
submission.to_csv(OUTPUT_PATH, index=False)

print(f"wrote {OUTPUT_PATH}  ({len(submission)} rows)")
print(f"  positives: {binary.sum()} ({binary.mean():.2%})")
print(f"  probability range: [{proba.min():.6f}, {proba.max():.6f}]")
print()
print(submission.head())

# %% [markdown]
# ## 10. Summary

# %%
print("=" * 66)
print(f"{'model':<28}{'OOF AUC':>12}{'composite':>14}")
print("-" * 66)
for r in results:
    print(f"{r['name']:<28}{r['auc']:>12.4f}{r['composite']:>14.4f}")
print("-" * 66)
print(f"{'BLEND (submitted)':<28}{final['auc']:>12.4f}{final['composite']:>14.4f}")
print("=" * 66)
print()
print("reference points:")
print(f"  {'all-ones baseline':<34}{0.4388:>10.4f}  <- must beat this")
print(f"  {'RandomForest (tuned threshold)':<34}{0.5167:>10.4f}")
print(f"  {'HistGradientBoosting':<34}{0.5269:>10.4f}")
print(f"  {'this notebook':<34}{final['composite']:>10.4f}")
print()
if final["composite"] < 0.4388:
    print("WARNING: below the all-ones floor — something is wrong, investigate before submitting.")
elif final["composite"] < 0.5269:
    print("Below the HistGradientBoosting reference. Check feature blocks and early stopping.")
else:
    print("Above both tree references. Expect CV to track the leaderboard closely")
    print("(train/test are iid, adversarial AUC 0.4985), so a large CV/LB gap means leakage.")
