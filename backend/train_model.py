"""
Train and save ML models for Urban Flood Risk prediction.
Dataset: flood.csv (50,000 rows, 20 features + FloodProbability)
Target: FloodRisk (LOW / MEDIUM / HIGH) derived from FloodProbability
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA_CANDIDATES = [
    os.path.join(ROOT, "data", "flood.csv"),
    r"C:\Users\PRISHA\Downloads\flood.csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if os.path.exists(p)), DATA_CANDIDATES[0])
MODELS_DIR = os.path.join(BASE, "models")
GRAPHS_DIR = os.path.join(ROOT, "public", "graphs")
RESULTS_DIR = os.path.join(BASE, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load & Label ─────────────────────────────────────────────────────────────
print("Loading dataset...")
print(f"  Path: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# Create risk label from FloodProbability
# Distribution: min=0.285, 25%=0.465, 50%=0.50, 75%=0.535, max=0.725
def label_risk(p):
    if p < 0.47:
        return "LOW"
    elif p <= 0.53:
        return "MEDIUM"
    else:
        return "HIGH"

df["FloodRisk"] = df["FloodProbability"].apply(label_risk)

label_counts = df["FloodRisk"].value_counts()
print(f"  Label distribution:\n{label_counts}")

FEATURES = [c for c in df.columns if c not in ["FloodProbability", "FloodRisk"]]
X = df[FEATURES].values
y = df["FloodRisk"].values

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ── Train Models ─────────────────────────────────────────────────────────────
print("\nTraining models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
}

metrics = {}
label_order = ["LOW", "MEDIUM", "HIGH"]

for name, model in models.items():
    print(f"  Training {name}...")
    Xtr = X_train_sc if name == "Logistic Regression" else X_train
    Xte = X_test_sc if name == "Logistic Regression" else X_test
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=label_order)

    metrics[name] = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "confusion_matrix": cm.tolist(),
        "labels": label_order,
    }
    print(f"    Accuracy: {acc*100:.2f}%")

# ── Save Best Model (Random Forest) ──────────────────────────────────────────
rf = models["Random Forest"]
joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(FEATURES, os.path.join(MODELS_DIR, "features.pkl"))
print(f"\nModels saved to {MODELS_DIR}")

# ── EDA Data ──────────────────────────────────────────────────────────────────
print("\nGenerating EDA data...")

# 1. FloodProbability distribution histogram (bucketed)
bins = np.arange(0.28, 0.73, 0.02)
counts, edges = np.histogram(df["FloodProbability"], bins=bins)
flood_prob_dist = [
    {"range": f"{e:.2f}", "count": int(c)}
    for e, c in zip(edges[:-1], counts)
]

# 2. Feature means by risk level
feature_means = df.groupby("FloodRisk")[FEATURES].mean().round(3).to_dict()

# 3. Correlation with FloodProbability (top 10)
correlations = df[FEATURES + ["FloodProbability"]].corr()["FloodProbability"].drop("FloodProbability")
top_corr = correlations.abs().nlargest(10).index.tolist()
corr_data = [
    {"feature": f, "correlation": round(float(correlations[f]), 4)}
    for f in top_corr
]

# 4. Risk label distribution
risk_dist = [
    {"label": k, "count": int(v), "pct": round(100 * v / len(df), 1)}
    for k, v in label_counts.items()
]

# 5. Feature importance from RF
importances = pd.Series(rf.feature_importances_, index=FEATURES)
top_imp = importances.nlargest(10)
feature_importance = [
    {"feature": f, "importance": round(float(v) * 100, 3)}
    for f, v in top_imp.items()
]

# 6. Per-feature box plots data (mean, std for chart)
feature_stats = []
for f in FEATURES:
    feature_stats.append({
        "feature": f,
        "mean": round(float(df[f].mean()), 2),
        "std": round(float(df[f].std()), 2),
        "min": int(df[f].min()),
        "max": int(df[f].max()),
        "corr": round(float(correlations[f]), 4),
    })

# Sort by abs corr
feature_stats.sort(key=lambda x: abs(x["corr"]), reverse=True)

# 7. Dataset summary
dataset_summary = {
    "total_records": len(df),
    "total_features": len(FEATURES),
    "missing_values": int(df.isnull().sum().sum()),
    "duplicates": int(df.duplicated().sum()),
    "low_risk_count": int(label_counts.get("LOW", 0)),
    "medium_risk_count": int(label_counts.get("MEDIUM", 0)),
    "high_risk_count": int(label_counts.get("HIGH", 0)),
    "flood_prob_mean": round(float(df["FloodProbability"].mean()), 4),
    "flood_prob_std": round(float(df["FloodProbability"].std()), 4),
}

# ── Statistical Tests ─────────────────────────────────────────────────────────
print("Running statistical tests...")

stat_results = []
for f in FEATURES:
    low_vals = df[df["FloodRisk"] == "LOW"][f].values
    high_vals = df[df["FloodRisk"] == "HIGH"][f].values
    t_stat, p_val = stats.ttest_ind(low_vals, high_vals)
    corr_coef, p_corr = stats.pearsonr(df[f].values, df["FloodProbability"].values)
    stat_results.append({
        "factor": f,
        "correlation": round(float(corr_coef), 4),
        "p_value": float(p_corr),
        "t_statistic": round(float(t_stat), 4),
        "t_p_value": float(p_val),
        "significant": bool(p_corr < 0.05),
    })

# Sort by abs corr
stat_results.sort(key=lambda x: abs(x["correlation"]), reverse=True)

# ── Generate Charts ───────────────────────────────────────────────────────────
print("Generating charts...")
sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "sans-serif"})

# 1. Correlation Heatmap (top 10 features + FloodProb)
top_feat_corr = [r["feature"] for r in corr_data]
corr_subset = df[top_feat_corr + ["FloodProbability"]].corr()
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.zeros_like(corr_subset, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = False  # show full matrix
sns.heatmap(
    corr_subset, annot=True, fmt=".2f", cmap="RdYlBu_r",
    ax=ax, linewidths=0.5, annot_kws={"size": 8},
    vmin=-0.3, vmax=0.3,
    cbar_kws={"label": "Pearson r"}
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "correlation_heatmap.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  Saved correlation_heatmap.png")

# 2. FloodProbability Distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["FloodProbability"], bins=50, color="#2563eb", alpha=0.8, edgecolor="white")
ax.axvline(x=0.47, color="#d97706", linestyle="--", linewidth=2, label="LOW|MED (0.47)")
ax.axvline(x=0.53, color="#dc2626", linestyle="--", linewidth=2, label="MED|HIGH (0.53)")
ax.set_xlabel("Flood Probability", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("FloodProbability Distribution", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "flood_probability.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  Saved flood_probability.png")

# 3. Top Features vs Flood Probability (scatter per top 4)
top4 = [r["feature"] for r in feature_stats[:4]]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
colors = {"LOW": "#10b981", "MEDIUM": "#d97706", "HIGH": "#dc2626"}
sample = df.sample(2000, random_state=42)
for i, feat in enumerate(top4):
    for risk, grp in sample.groupby("FloodRisk"):
        axes[i].scatter(grp[feat], grp["FloodProbability"],
                        alpha=0.3, s=8, label=risk, color=colors[risk])
    axes[i].set_xlabel(feat, fontsize=10)
    axes[i].set_ylabel("FloodProbability", fontsize=10)
    axes[i].set_title(f"{feat} vs Flood Probability", fontsize=11, fontweight="bold")
    if i == 0:
        axes[i].legend(fontsize=8)
plt.suptitle("Key Features vs Flood Probability", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "features_vs_flood.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  Saved features_vs_flood.png")

# 4. Feature Importance Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
imp_df = pd.DataFrame(feature_importance).sort_values("importance")
ax.barh(imp_df["feature"], imp_df["importance"], color="#2563eb", alpha=0.85)
ax.set_xlabel("Importance (%)", fontsize=12)
ax.set_title("Random Forest – Feature Importance (Top 10)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "feature_importance.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  Saved feature_importance.png")

# 5. Risk Distribution Pie
fig, ax = plt.subplots(figsize=(7, 7))
pie_colors = ["#10b981", "#d97706", "#dc2626"]
pie_labels = []
pie_sizes = []
for r in ["LOW", "MEDIUM", "HIGH"]:
    row = next((x for x in risk_dist if x["label"] == r), None)
    if row:
        pie_labels.append(f'{r}\n({row["pct"]}%)')
        pie_sizes.append(row["count"])
ax.pie(pie_sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
       startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Flood Risk Label Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "risk_distribution.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  Saved risk_distribution.png")

# 6. Confusion Matrix for Random Forest
cm_rf = np.array(metrics["Random Forest"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm_rf, annot=True, fmt="d", cmap="Blues",
    xticklabels=label_order, yticklabels=label_order,
    ax=ax, linewidths=0.5, linecolor="white"
)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Confusion Matrix – Random Forest", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "confusion_matrix_rf.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  Saved confusion_matrix_rf.png")

# 7. MonsoonIntensity vs FloodProbability Box
fig, ax = plt.subplots(figsize=(10, 5))
sample_box = df.sample(5000, random_state=42)
sample_box.boxplot(column="FloodProbability", by="MonsoonIntensity", ax=ax,
                   patch_artist=True, medianprops=dict(color="red", linewidth=2))
ax.set_xlabel("Monsoon Intensity (0-16)", fontsize=12)
ax.set_ylabel("Flood Probability", fontsize=12)
ax.set_title("Monsoon Intensity vs Flood Probability", fontsize=13, fontweight="bold")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "monsoon_vs_flood.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  Saved monsoon_vs_flood.png")

# 8. Urbanization vs FloodProbability Box
fig, ax = plt.subplots(figsize=(10, 5))
df.boxplot(column="FloodProbability", by="Urbanization", ax=ax,
           patch_artist=True, medianprops=dict(color="red", linewidth=2))
ax.set_xlabel("Urbanization (0-17)", fontsize=12)
ax.set_ylabel("Flood Probability", fontsize=12)
ax.set_title("Urbanization vs Flood Probability", fontsize=13, fontweight="bold")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "urbanization_vs_flood.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  Saved urbanization_vs_flood.png")

# ── Save JSON results ─────────────────────────────────────────────────────────
results = {
    "dataset_summary": dataset_summary,
    "model_metrics": metrics,
    "eda": {
        "flood_prob_distribution": flood_prob_dist,
        "risk_distribution": risk_dist,
        "feature_stats": feature_stats,
        "feature_importance": feature_importance,
        "top_correlations": corr_data,
    },
    "statistical_tests": stat_results,
    "features": FEATURES,
}

out_path = os.path.join(RESULTS_DIR, "analysis_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll results saved to {out_path}")
print("Training complete!")
