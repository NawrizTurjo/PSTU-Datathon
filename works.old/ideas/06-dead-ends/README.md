# 06 — Dead ends (measured)

Things that look promising, are commonly suggested for this kind of competition, and
**were tested on this dataset and do not work**. Ten minutes of reading here saves hours.

---

## 1. Exact-match train→test lookup — **tested, actively harmful**

**The idea:** 7.3% of test rows (841) are byte-identical to a train row on all 286
features. Surely the matched train row's label is a free answer for those?

**Why it fails:** the model is *better* than the lookup on exactly the rows the lookup
covers.

Measured via 5-fold OOF on train (matching each validation fold against its training
folds only — the honest simulation of the train→test situation):

| | AUC on the 3,403 matched rows |
|---|---|
| Model prediction | **0.8295** |
| Duplicate-group mean lookup | **0.7210** |

Blending the lookup in makes the score monotonically worse:

| Blend weight on lookup | OOF AUC | Best composite |
|---|---|---|
| 0.00 (baseline) | 0.8189 | **0.5269** |
| 0.15 | 0.8186 | 0.5262 |
| 0.30 | 0.8177 | 0.5248 |
| 0.50 | 0.8153 | 0.5189 |
| 1.00 (full override) | 0.7889 | **0.5005** |

A full override costs **−0.026 composite**.

**Why the intuition is wrong:** the matched rows are overwhelmingly the sparse
"nothing happened this period" default profile — the largest duplicate group has 430 rows
with 219 of its 286 features exactly zero. These rows collide because they're
*uninformative*, not because they're the same station. The group mean is a coarse
5–7% base-rate estimate, while the model can still discriminate within them using the
handful of features that do vary.

Using it as an engineered feature instead of an override gave +0.0015 composite — and
even that number is optimistic, since it was computed with in-fold group means (i.e. mild
leakage). Not worth the leakage risk.

**Verdict: skip entirely.** Note this corrects an earlier suggestion in
`dataset_exploration/README.md`, which proposed exploiting this overlap before it was
measured.

---

## 2. GroupKFold on pseudo-station ids — **no such grouping exists**

**The idea:** there's no `station_id`, but stations could be recovered by grouping on
static `base_*` attributes, and then `GroupKFold` would be needed to prevent leakage
across folds.

**Why it fails:** the apparent groups are artifacts.

- 19% of train rows fall into a `base_*` group of size >1, and the largest has 1,265 rows —
  superficially convincing.
- But `base_distance_from_coastal_river_km` has a single dominant fill value
  (`0.5090757718457412`) covering **19.5% of all rows**, and
  `base_solar_panel_tilt_angle_degrees` has only **5 distinct values**. Unrelated rows
  collide on these constants.
- Decisive test: within-group target rates scatter around the global 5% rate with a
  standard deviation of **0.113**. A real station id would produce groups clustering
  tightly around each station's own risk level. These don't.

**Verdict: use `StratifiedKFold`.** `GroupKFold` on these keys would just shrink your
effective training data for no reason.

---

## 3. Adversarial validation / covariate-shift correction — **nothing to correct**

**The idea:** reweight training samples, or drop features, to compensate for train/test
distribution shift.

**Why it fails:** there is no shift. A 5-fold RandomForest trained to distinguish train
rows from test rows achieves **ROC-AUC 0.4985** — indistinguishable from chance.

A handful of numeric columns have test values marginally outside train's observed
`[min, max]` (33 of 223 columns, typically 1–2 rows past the boundary), and boolean flag
rates differ by under 0.5 percentage points. None of that constitutes shift worth modelling.

**Verdict: skip.** Useful corollary — since train and test are iid, **your CV should track
the leaderboard closely.** A large CV/LB gap means you introduced leakage, not that the
split is adversarial.

---

## 4. Hunting for more sentinel values — **there's only one**

**The idea:** Santander-derived data often hides several missing-value codes (`-1`, `99`,
`999`, `9999999999`).

**Why it fails:** an exhaustive scan of every numeric column for all the standard codes
found only `-999999` in `base_number_of_dependent_farmers` (66 train / 23 test rows).
The value `99` appears in some `num_var*` columns at under 0.15% of rows each, but those
columns range into the hundreds, so 99 is a plausible genuine count — not sentinel evidence.

**Verdict: handle `-999999`, stop looking.** It affects 66 rows and is a correctness
detail, not a scoring lever.

---

## 5. Santander de-anonymization — **no reliable mapping exists**

**The idea:** the `num_var*` / `num_op_var*` columns come from Santander Customer
Satisfaction; reverse-engineer their original banking meanings to build better features.

**Why it fails:** beyond `var3` (a country code carrying the `-999999` sentinel — which
maps here to `base_number_of_dependent_farmers`, already handled) and community
speculation about `num_var38` as a balance figure, there is no verified public
field-by-field mapping. More importantly, this competition **reassigned its own synthetic
semantics** to those columns. The `sensor_*` / `cost_*` / `count_*` names are the real
meanings for this data; Santander's original meanings don't transfer.

**Verdict: skip the archaeology.** What *does* transfer from Santander is the
**modelling technique** — specifically the row-wise aggregate features in
[../02-feature-engineering/](../02-feature-engineering/) Block A, which work because of
the shared sparsity structure regardless of what the columns mean.

---

## 6. Optimizing the threshold with Nelder-Mead / Bayesian search — **overkill**

The threshold search is **one-dimensional and bounded on (0,1)**. A dense grid
(200 points, milliseconds) is exhaustive and cannot get stuck. Derivative-free optimizers
add dependencies and failure modes for zero benefit.

**Verdict: use a grid.** These methods only become relevant if you jointly optimize blend
weights *and* threshold in one multi-dimensional search.

---

## Things NOT on this list

These remain genuinely open, untested, and worth your time:

- Santander-style row aggregates ([02, Block A](../02-feature-engineering/)) — strong prior, unverified here
- Domain ratio features ([02, Blocks C/D](../02-feature-engineering/)) — plausible, unverified
- CatBoost vs LightGBM head-to-head ([01](../01-gbdt-core/))
- Sample weighting for label noise ([05](../05-label-noise/))
- Class-weighting A/B on OOF AUC ([01](../01-gbdt-core/))
