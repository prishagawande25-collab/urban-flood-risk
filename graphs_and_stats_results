# ============================================================
# URBAN FLOOD RISK PREDICTION
# Person 2 - EDA + Statistical Analysis
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("/content/flood.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nBasic Statistics:")
print(df.describe())


# ============================================================
# 2. FLOOD PROBABILITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["FloodProbability"],
    bins=30,
    kde=True
)

plt.title("Distribution of Flood Probability")
plt.xlabel("Flood Probability")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    "flood_probability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 3. MONSOON INTENSITY VS FLOOD PROBABILITY
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="MonsoonIntensity",
    y="FloodProbability"
)

plt.title("Monsoon Intensity vs Flood Probability")
plt.xlabel("Monsoon Intensity")
plt.ylabel("Flood Probability")
plt.tight_layout()

plt.savefig(
    "monsoon_vs_flood.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 4. URBANIZATION VS FLOOD PROBABILITY
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Urbanization",
    y="FloodProbability"
)

plt.title("Urbanization vs Flood Probability")
plt.xlabel("Urbanization")
plt.ylabel("Flood Probability")
plt.tight_layout()

plt.savefig(
    "urbanization_vs_flood.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 5. DRAINAGE SYSTEMS VS FLOOD PROBABILITY
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="DrainageSystems",
    y="FloodProbability"
)

plt.title("Drainage Systems vs Flood Probability")
plt.xlabel("Drainage Systems")
plt.ylabel("Flood Probability")
plt.tight_layout()

plt.savefig(
    "drainage_vs_flood.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 6. DEFORESTATION VS FLOOD PROBABILITY
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Deforestation",
    y="FloodProbability"
)

plt.title("Deforestation vs Flood Probability")
plt.xlabel("Deforestation")
plt.ylabel("Flood Probability")
plt.tight_layout()

plt.savefig(
    "deforestation_vs_flood.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 7. DETERIORATING INFRASTRUCTURE VS FLOOD PROBABILITY
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="DeterioratingInfrastructure",
    y="FloodProbability"
)

plt.title("Deteriorating Infrastructure vs Flood Probability")
plt.xlabel("Deteriorating Infrastructure")
plt.ylabel("Flood Probability")
plt.tight_layout()

plt.savefig(
    "infrastructure_vs_flood.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 8. CORRELATION ANALYSIS
# ============================================================

correlation = df.corr(numeric_only=True)

print("\n--- CORRELATION WITH FLOOD PROBABILITY ---")

print(
    correlation["FloodProbability"]
    .sort_values(ascending=False)
)


# ============================================================
# 9. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(16, 12))

sns.heatmap(
    correlation,
    annot=True,
    cmap="magma",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Flood Risk Factors")
plt.tight_layout()

plt.savefig(
    "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. RANK FACTORS BY CORRELATION
# ============================================================

correlation_with_flood = (
    correlation["FloodProbability"]
    .drop("FloodProbability")
    .sort_values(
        key=abs,
        ascending=False
    )
)

print("\n--- FACTORS RANKED BY CORRELATION ---")
print(correlation_with_flood)


# ============================================================
# 11. CREATE CORRELATION RESULTS CSV
# ============================================================

correlation_results_df = correlation_with_flood.reset_index()

correlation_results_df.columns = [
    "Factor",
    "Correlation"
]

print("\n--- CORRELATION RESULTS CSV ---")
print(correlation_results_df)

correlation_results_df.to_csv(
    "correlation_results.csv",
    index=False
)


# ============================================================
# 12. PEARSON STATISTICAL TESTS
# ============================================================

variables = [
    "MonsoonIntensity",
    "Urbanization",
    "DrainageSystems",
    "Deforestation",
    "DeterioratingInfrastructure"
]

results = []

for col in variables:

    r, p = pearsonr(
        df[col],
        df["FloodProbability"]
    )

    print(f"\n{col}:")
    print(f"  Correlation = {r:.3f}")
    print(f"  P-value = {p:.5f}")

    results.append({
        "Factor": col,
        "Correlation": r,
        "P_Value": p
    })


# ============================================================
# 13. CREATE STATISTICAL RESULTS CSV
# ============================================================

results_df = pd.DataFrame(results)

print("\n--- STATISTICAL RESULTS CSV ---")
print(results_df)

results_df.to_csv(
    "statistical_results.csv",
    index=False
)


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\n================================================")
print("EDA AND STATISTICAL ANALYSIS COMPLETED")
print("================================================")

print("\nGenerated 7 PNG files:")
print("1. flood_probability.png")
print("2. monsoon_vs_flood.png")
print("3. urbanization_vs_flood.png")
print("4. drainage_vs_flood.png")
print("5. deforestation_vs_flood.png")
print("6. infrastructure_vs_flood.png")
print("7. correlation_heatmap.png")

print("\nGenerated 2 CSV files:")
print("8. correlation_results.csv")
print("9. statistical_results.csv")
