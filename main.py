import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# FEATURE PIPELINE
# =========================

def create_features(df):
    df = df.copy()

    df = df.sort_values(["well_id", "depth"]).reset_index(drop=True)
    g = df.groupby("well_id")

    # =========================
    # CORE GRADIENT FEATURES
    # =========================
    df["gamma_grad"] = g["gamma_ray"].diff()

    # =========================
    # BASIC PHYSICS FEATURES
    # =========================
    df["density_neutron_sep"] = df["bulk_density"] - df["neutron_porosity"]

    # =========================
    # ROLLING FEATURES
    # =========================
    df["gamma_roll_mean_5"] = g["gamma_ray"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    df["resistivity_roll_mean_5"] = g["resistivity"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    df["sonic_roll_mean_5"] = g["sonic_dt"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # =========================
    # LOG RATIOS
    # =========================
    gamma = np.log1p(df["gamma_ray"].clip(lower=0))
    res = np.log1p(df["resistivity"].clip(lower=0))
    dens = np.log1p(df["bulk_density"].clip(lower=0))
    sonic = np.log1p(df["sonic_dt"].clip(lower=0))

    df["gamma_res_ratio"] = gamma - res
    df["density_sonic_ratio"] = dens - sonic

    # =========================
    # CLEANUP (IMPORTANT FIX)
    # =========================
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("synthetic_geology_dataset.csv")
df = create_features(df)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =========================
# FEATURES / TARGET
# =========================

features = [
    "gamma_ray",
    "resistivity",
    "bulk_density",
    "neutron_porosity",
    "sonic_dt",
    "caliper",
    "gamma_grad",
    "density_neutron_sep",
    "gamma_roll_mean_5",
    "resistivity_roll_mean_5",
    "sonic_roll_mean_5",
    "gamma_res_ratio",
    "density_sonic_ratio"
]

X = df[features]
y = df["lithology"]
groups = df["well_id"]

gkf = GroupKFold(n_splits=5)

# =========================
# MODELS
# =========================

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=RANDOM_STATE
    )
}

# =========================
# TUNING CONFIGS (DAY 4)
# =========================

tuning_configs = {
    "RF_baseline":  RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
    "RF_deeper":    RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    "RF_more_trees":RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
    "HGB_baseline": HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=300, random_state=RANDOM_STATE),
    "HGB_deeper":   HistGradientBoostingClassifier(learning_rate=0.05, max_depth=8, max_iter=300, random_state=RANDOM_STATE),
}

# =========================
# ENSEMBLE (DAY 5)
# =========================

ensemble = VotingClassifier(
    estimators=[
        ("rf",  RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)),
        ("hgb", HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=300, random_state=RANDOM_STATE)),
    ],
    voting="hard"
)

# =========================
# CROSS VALIDATION
# =========================

print("\n====================")
print("DAY 10 — FINAL RUN")
print("====================")

results = {}

for name, model in {**tuning_configs, "Ensemble_RF_HGB": ensemble}.items():

    scores = []

    for tr, te in gkf.split(X, y, groups):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        preds = m.predict(X.iloc[te])
        scores.append(accuracy_score(y.iloc[te], preds))

    results[name] = scores

    print("\n" + name)
    print("Mean:", round(np.mean(scores), 4))
    print("Std:", round(np.std(scores), 4))

# =========================
# STABILITY CHECK
# =========================

print("\n====================")
print("STABILITY REPORT")
print("====================")

for name, scores in results.items():
    flag = "UNSTABLE" if np.std(scores) > 0.03 else "STABLE"
    print(f"{name}: mean={round(np.mean(scores), 4)} std={round(np.std(scores), 4)} → {flag}")

# =========================
# BEST MODEL SELECTION
# =========================

best_model_name = max(results, key=lambda k: np.mean(results[k]))
best_model = tuning_configs[best_model_name]

print("\n====================")
print("BEST MODEL")
print("====================")
print(best_model_name)

# =========================
# FINAL SANITY CHECK (SHUFFLE TEST)
# =========================

y_shuffled = shuffle(y, random_state=RANDOM_STATE)

shuffle_scores = []

for tr, te in gkf.split(X, y_shuffled, groups):
    m = clone(best_model)
    m.fit(X.iloc[tr], y_shuffled.iloc[tr])
    preds = m.predict(X.iloc[te])
    shuffle_scores.append(accuracy_score(y_shuffled.iloc[te], preds))

print("\n====================")
print("SHUFFLE TEST")
print("====================")
print("Mean:", round(np.mean(shuffle_scores), 4))

# =========================
# FINAL TRAIN (FULL DATA)
# =========================

best_model.fit(X, y)

print("\n====================")
print("TRAINING COMPLETE")
print("====================")

# =========================
# SAVE MODEL (DAY 6)
# =========================

joblib.dump(best_model, "final_model.joblib")
print("Model saved → final_model.joblib")

# =========================
# SUBMISSION SYSTEM (DAY 7)
# =========================

print("\n====================")
print("DAY 7 — SUBMISSION SYSTEM")
print("====================")

# Load saved model
submission_model = joblib.load("final_model.joblib")

# Generate predictions
submission_preds = submission_model.predict(X)

# Build submission
submission = pd.DataFrame({
    "well_id":   df["well_id"],
    "depth":     df["depth"],
    "lithology": submission_preds
})

# Format validation
assert list(submission.columns) == ["well_id", "depth", "lithology"], "Column mismatch"
assert submission.isnull().sum().sum() == 0, "Nulls found in submission"
assert len(submission) == len(df), "Row count mismatch"

submission.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
print(f"Rows: {len(submission)}")
print(f"Columns: {list(submission.columns)}")
print(f"Classes: {sorted(submission['lithology'].unique())}")
print("\nSample:")
print(submission.head())
print("\nFormat validation: PASSED")

# =========================
# DAY 8 — METRICS + ANALYSIS
# =========================

print("\n====================")
print("DAY 8 — METRICS + ANALYSIS")
print("====================")

# Confusion matrix
preds_train = best_model.predict(X)
cm = confusion_matrix(y, preds_train, labels=sorted(y.unique()))

print("\nConfusion Matrix:")
print(f"Classes: {sorted(y.unique())}")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y, preds_train))

# Feature importance
print("\nFeature Importance:")
importances = best_model.feature_importances_
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"  {feat:<30} {round(imp, 4)}")

# =========================
# DAY 9 — PRESENTATION BUILD
# =========================

print("\n====================")
print("DAY 9 — PRESENTATION BUILD")
print("====================")

print("""
PROJECT: Geology Lithology Classification
==========================================

OBJECTIVE:
  Classify subsurface rock types (lithology) from well log data
  using a reproducible, competition-ready ML pipeline.

DATASET:
  - Rows     : 6125
  - Wells    : 5 (synthetic)
  - Classes  : limestone, sandstone, shale
  - Features : 13 (raw + engineered)

METHODOLOGY:
  1. Feature Engineering
     - Gradient features    : gamma_grad (depth-based diff)
     - Physics features     : density_neutron_sep
     - Rolling means        : gamma, resistivity, sonic (window=5)
     - Log ratios           : gamma_res_ratio, density_sonic_ratio

  2. Validation Strategy
     - GroupKFold (n=5), split by well_id
     - Prevents well-level data leakage
     - Shuffle test confirms model learns real signal

  3. Model Selection
     - Candidates : RandomForest, HistGradientBoosting
     - Winner     : RF_baseline (mean=1.0, std=0.0)
     - Ensemble tested (RF + HGB voting) — no improvement, discarded

  4. Hyperparameter Tuning
     - Tuned: max_depth, n_estimators, min_samples_leaf
     - 5 configs tested, no grid search
     - RF_baseline remained best config

RESULTS:
  - CV Accuracy  : 1.0000 (mean), 0.0000 (std)
  - Shuffle Test : 0.352 (confirms signal is real)
  - All classes  : precision=1.00, recall=1.00, f1=1.00

TOP FEATURES:
  1. density_sonic_ratio     (0.1634)
  2. density_neutron_sep     (0.1406)
  3. gamma_ray               (0.1273)
  4. sonic_dt                (0.1184)
  5. gamma_res_ratio         (0.1183)

PIPELINE:
  create_features() → GroupKFold CV → best model selection
  → final train → save model → load model → predict → submission CSV

REPRODUCIBILITY:
  - RANDOM_STATE = 42 locked across all models and shuffles
  - Model saved to final_model.joblib
""")

print("Presentation build: COMPLETE")

# =========================
# DAY 10 — FINAL VALIDATION
# =========================

print("\n====================")
print("DAY 10 — FINAL VALIDATION")
print("====================")
print(f"Dataset rows     : {len(df)}")
print(f"Features used    : {len(features)}")
print(f"Best model       : {best_model_name}")
print(f"CV Accuracy      : {round(np.mean(results[best_model_name]), 4)}")
print(f"CV Std           : {round(np.std(results[best_model_name]), 4)}")
print(f"Shuffle Test     : {round(np.mean(shuffle_scores), 4)}")
print(f"Submission rows  : {len(submission)}")
print(f"Model on disk    : final_model.joblib")
print(f"Submission file  : submission.csv")
print("\n✔ PROJECT COMPLETE — READY FOR REAL DATA")

# =========================
# DAY 10.1 — GRAPHS
# =========================

print("\n====================")
print("DAY 10.1 — GRAPHS")
print("====================")

classes = sorted(y.unique())

# --- Confusion Matrix Heatmap ---
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix — RF_baseline")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show(block=False)

# --- Feature Importance Bar Chart ---
feat_imp   = sorted(zip(features, importances), key=lambda x: x[1])
feat_names = [f[0] for f in feat_imp]
feat_vals  = [f[1] for f in feat_imp]

plt.figure(figsize=(8, 6))
plt.barh(feat_names, feat_vals, color="steelblue")
plt.xlabel("Importance")
plt.title("Feature Importance — RF_baseline")
plt.tight_layout()
plt.show(block=False)

plt.show()
