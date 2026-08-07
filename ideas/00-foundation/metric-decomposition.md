# What the scoring formula actually rewards

This is the highest-value document in this folder. The competition metric is unusual,
and understanding its structure tells you where the points are.

## The official formula

$$\text{Score} = 0.30 F_1 + 0.25 \text{AUC} + 0.15 P + 0.15 R + 0.10 \text{BalAcc} + 0.05 S$$

where $P$ = precision, $R$ = recall, $S$ = specificity, $\text{BalAcc} = (R+S)/2$.

## It collapses to five terms

Since $\text{BalAcc}$ is just $(R+S)/2$, substitute it:

$$0.10 \cdot \frac{R+S}{2} = 0.05R + 0.05S$$

which gives an exactly equivalent, simpler form:

$$\boxed{\text{Score} = 0.30 F_1 + 0.25\,\text{AUC} + 0.15 P + 0.20 R + 0.10 S}$$

**This is verified numerically** — the two forms agree to machine precision at every
threshold tested. Use the collapsed form when reasoning; use either when coding.

### What this reveals

| Term | Effective weight | Notes |
|---|---|---|
| $F_1$ | 0.30 | balances P and R against each other |
| AUC | 0.25 | **threshold-free** — pure ranking quality |
| $R$ (recall) | **0.20** | |
| $P$ (precision) | **0.15** | |
| $S$ (specificity) | 0.10 | cheap: 95% of rows are negative |

**Recall is weighted higher than precision** (0.20 vs 0.15) once BalAcc is unpacked —
the raw formula hides this by splitting recall's weight across two terms. The $F_1$
term pulls back toward balance, but the net tilt still favours catching failures over
avoiding false alarms. This matches the competition's stated intent, and it means the
optimal threshold sits **below** the point that maximizes $F_1$ alone.

## Two independent levers

The submission has two separate columns, and they feed disjoint parts of the score:

- **`Target_Probability`** → feeds **only** AUC (0.25 of the score).
- **`Target_Binary`** → feeds **everything else** (0.75 of the score).

They are scored independently, so optimize them independently:

1. Make the probability column the **best-ranked** scores you have (rank-averaged
   ensemble is ideal — see [../04-ensemble-diversity/](../04-ensemble-diversity/)).
2. Choose the binary column to maximize the remaining 0.75, which is a **one-dimensional
   search** over how many rows you label positive.

There is no requirement that the binary column be a threshold of the probability column —
though in practice, for a fixed number of predicted positives, taking the top-$k$ by
probability is optimal, so a threshold search is the right method.

## Degenerate baselines (measured)

| Submission | $F_1$ | $P$ | $R$ | $S$ | Composite* |
|---|---|---|---|---|---|
| all zeros | 0.0000 | 0.0000 | 0.0000 | 1.0000 | **0.3028** |
| all ones | 0.0952 | 0.0500 | 1.0000 | 0.0000 | **0.4388** |

\* assuming an AUC of 0.811 from the probability column.

**All-ones beats all-zeros by a wide margin (0.4388 vs 0.3028).** This is a direct
consequence of the recall tilt above. Two implications:

- Any model scoring below 0.4388 is worse than a trivial constant submission. Use this
  as a sanity floor.
- When in doubt at the margin, **err toward predicting more positives**, not fewer.
  The metric punishes timidity far more than over-alerting.

## Where the score actually comes from

At the measured optimum for a HistGradientBoosting baseline (AUC 0.8189, threshold 0.53):

```
AUC term            0.25 × 0.8189 = 0.2047   (of a possible 0.25)
threshold-dependent part          = 0.3222   (of a possible 0.75)
                            total = 0.5269
```

The threshold-dependent 0.75 is where the headroom is — but it is **gated by ranking
quality**. At that optimum:

```
F1 = 0.2829    Precision = 0.1830    Recall = 0.6226    Specificity = 0.8537
```

**Precision is the binding constraint.** At 5% base rate, catching 62% of failures means
82% of your alerts are false. You cannot fix this with thresholding — only by ranking
better. This is why improving AUC pays twice: once directly (0.25 weight), and again by
raising the whole precision/recall frontier the threshold search operates on.

## Practical consequences

1. **Never submit a 0.5-threshold binary column.** Measured cost: ~0.018 composite.
2. **Re-tune the threshold for every model.** The optimum moved from 0.60 (RandomForest)
   to 0.53 (HistGradientBoosting) — it tracks model calibration, not some universal value.
3. **Calibration does not change AUC.** Isotonic/Platt scaling is monotone, so it cannot
   alter the ranking or the AUC term. It is worth doing only to make the threshold more
   stable and transferable between CV and test — not as a score improvement in itself.
   (Note the distinction: *class weighting during training* changes the fitted model and
   therefore **can** change AUC. Post-hoc calibration cannot.)
4. **Rank-average, don't probability-average, for the AUC column.** Only the order matters.
5. **Clip probabilities away from exact 0.0 and 1.0** — the submission spec requires values
   strictly inside the range and will reject the file otherwise.

## Reusable scoring function

```python
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, confusion_matrix)

def composite_score(y_true, y_pred_binary, y_proba):
    """Exact competition metric. Returns the score and its components."""
    f1   = f1_score(y_true, y_pred_binary, zero_division=0)
    auc  = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec  = recall_score(y_true, y_pred_binary, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    bal  = (rec + spec) / 2
    score = 0.30*f1 + 0.25*auc + 0.15*prec + 0.15*rec + 0.10*bal + 0.05*spec
    return score, dict(f1=f1, auc=auc, precision=prec, recall=rec,
                       specificity=spec, balanced_accuracy=bal)
```

Use this — not `sklearn`'s `f1_score` alone, and not accuracy — as the selection
criterion for every hyperparameter, feature block, and blend weight you evaluate.
