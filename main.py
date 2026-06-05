import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, GradientBoostingClassifier

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("synthetic_geology_dataset.csv")
df = df.sort_values(["well_id", "depth"]).reset_index(drop=True)

np.random.seed(42)

# =========================
# BASIC FEATURE ENGINEERING
# =========================

df["gamma_grad"] = df.groupby("well_id")["gamma_ray"].diff()

df["density_neutron_sep"] = df["bulk_density"] - df["neutron_porosity"]

df["gamma_roll_mean_5"] = df.groupby("well_id")["gamma_ray"].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)

df["resistivity_roll_mean_5"] = df.groupby("well_id")["resistivity"].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)

df["sonic_roll_mean_5"] = df.groupby("well_id")["sonic_dt"].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)

df["gamma_res_ratio"] = np.log1p(df["gamma_ray"]) - np.log1p(df["resistivity"])
df["density_sonic_ratio"] = np.log1p(df["bulk_density"]) - np.log1p(df["sonic_dt"])

# =========================
# CLEAN DATA
# =========================

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)

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
        random_state=42,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingClassifier(random_state=42),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=42
    )
}

# =========================
# CROSS VALIDATION
# =========================

print("\n====================")
print("PHASE 3.3 FINAL VALIDATION")
print("====================")

results = {}

for name, model in models.items():

    scores = []

    for tr, te in gkf.split(X, y, groups):

        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])

        scores.append(accuracy_score(y.iloc[te], preds))

    results[name] = scores

    print("\n" + name)
    print("Mean:", round(np.mean(scores), 4))
    print("Std:", round(np.std(scores), 4))

# =========================
# BEST MODEL SELECTION
# =========================

best_model_name = max(results, key=lambda k: np.mean(results[k]))
best_model = models[best_model_name]

print("\n====================")
print("BEST MODEL")
print("====================")
print(best_model_name)

# =========================
# FINAL SANITY CHECK (SHUFFLE TEST)
# =========================

from sklearn.utils import shuffle

y_shuffled = shuffle(y, random_state=42)

shuffle_scores = []

for tr, te in gkf.split(X, y_shuffled, groups):
    best_model.fit(X.iloc[tr], y_shuffled.iloc[tr])
    preds = best_model.predict(X.iloc[te])
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

print("Final model trained on full dataset.")

# =========================
# PHASE 4 - SUBMISSION MODE
# =========================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("\n====================")
print("PHASE 4 - SUBMISSION MODE")
print("====================")

# Ensure features match training
X_full = X.copy()
y_full = y.copy()

# Train final model (use best from Phase 3.3)
final_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_full, y_full)

# Predict probabilities (optional but safer for Kaggle)
preds = final_model.predict(X_full)

# Build submission format
submission = pd.DataFrame({
    "well_id": df["well_id"],
    "depth": df["depth"],
    "lithology": preds
})

# Save file
submission.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
print(submission.head())
