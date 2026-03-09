import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# -------------------------------------------------------
# Loading out-of-sample prediction file
# -------------------------------------------------------

candidates = [
    ("oos_predictions_policy_calibrated_hier.csv", "p_final_cal"),
    ("oos_predictions_policy.csv", "p_final"),
    ("oos_predictions.csv", "p_hybrid"),
]

for file_path, pred_col in candidates:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        prediction_column = pred_col
        break
else:
    raise FileNotFoundError("No out-of-sample prediction file found.")

# -------------------------------------------------------
# Data preparation
# -------------------------------------------------------

required_cols = ["y_true", prediction_column]
df = df.dropna(subset=required_cols).copy()
df["y_true"] = df["y_true"].astype(int)

for col in ["p_ucm", "p_ml", "p_hybrid", "p_final", "p_final_cal"]:
    if col in df.columns:
        df[col] = np.clip(df[col].astype(float), 1e-6, 1 - 1e-6)

if "p_final_cal" not in df.columns:
    df["p_final_cal"] = np.clip(df[prediction_column].astype(float), 1e-6, 1 - 1e-6)

os.makedirs("figures", exist_ok=True)


def add_roc_curve(ax, y, scores, label):
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")

def add_pr_curve(ax, y, scores, label):
    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")

def reliability_curve(data, prob_col, bins=10):
    tmp = data[[prob_col, "y_true"]].copy()
    tmp["bin"] = pd.qcut(tmp[prob_col], q=min(bins, tmp[prob_col].nunique()), duplicates="drop")
    summary = (
        tmp.groupby("bin", observed=True)
        .agg(
            predicted=(prob_col, "mean"),
            observed=("y_true", "mean"),
            n=("y_true", "size"),
        )
        .reset_index(drop=True)
    )
    return summary

# -------------------------------------------------------
# Figure 1: Overall ROC
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

for col, label in [
    ("p_ucm", "UCM"),
    ("p_ml", "ML"),
    ("p_hybrid", "Hybrid"),
    ("p_final", "Base"),
    ("p_final_cal", "Calibrated"),
]:
    if col in df.columns:
        add_roc_curve(ax, df["y_true"], df[col], label)

ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Overall ROC (pooled out-of-sample)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/01_overall_roc.png", dpi=200)
plt.close(fig)

# -------------------------------------------------------
# Figure 2: Overall Precision–Recall
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))
prevalence = df["y_true"].mean()

for col, label in [
    ("p_ucm", "UCM"),
    ("p_ml", "ML"),
    ("p_hybrid", "Hybrid"),
    ("p_final", "Base"),
    ("p_final_cal", "Calibrated"),
]:
    if col in df.columns:
        add_pr_curve(ax, df["y_true"], df[col], label)

ax.axhline(prevalence, linestyle="--", color="gray", lw=1, label=f"Baseline={prevalence:.2f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Overall Precision–Recall (pooled out-of-sample)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/02_overall_pr.png", dpi=200)
plt.close(fig)

# -------------------------------------------------------
# Figure 3: Overall Calibration
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect")

if "p_final" in df.columns:
    uncal = reliability_curve(df, "p_final", bins=10)
    ax.plot(uncal["predicted"], uncal["observed"], marker="o", label="Uncalibrated")
    for px, py, n in zip(uncal["predicted"], uncal["observed"], uncal["n"]):
        ax.text(px, py, f"n={n}", fontsize=8)

cal = reliability_curve(df, "p_final_cal", bins=10)
ax.plot(cal["predicted"], cal["observed"], marker="o", label="Calibrated")
for px, py, n in zip(cal["predicted"], cal["observed"], cal["n"]):
    ax.text(px, py, f"n={n}", fontsize=8)

ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed event rate")
ax.set_title("Calibration (pooled out-of-sample)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/03_overall_calibration.png", dpi=200)
plt.close(fig)