import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

TRAIN_DIR = "/kaggle/input/rogii-wellbore-geology-prediction/train/"
TEST_DIR  = "/kaggle/input/rogii-wellbore-geology-prediction/test/"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FEATURES = [
    "MD", "X", "Y", "Z",
    "GR", "GR_roll_mean_10", "GR_roll_mean_30", "GR_grad", "GR_roll_std_10",
    "TVT_last_known", "TVT_known_grad", "TVT_known_grad2",
    "MD_diff", "Z_diff", "Z_grad2",
    "ANCC_filled", "dist_to_EGFDL", "dist_to_BUDA", "dist_to_ASTNU",
    "dist_to_ASTNL", "dist_to_EGFDU",
    "GR_x_TVT", "Z_x_TVT"
]

def load_wells(directory):
    horizontal_files = sorted(glob.glob(os.path.join(directory, "*__horizontal_well.csv")))
    wells = []
    for hf in horizontal_files:
        well_name = os.path.basename(hf).replace("__horizontal_well.csv", "")
        h_df = pd.read_csv(hf)
        h_df["WELLNAME"] = well_name
        wells.append(h_df)
    return pd.concat(wells, ignore_index=True)

def create_features(df):
    df = df.copy()
    df = df.sort_values(["WELLNAME", "MD"]).reset_index(drop=True)
    g = df.groupby("WELLNAME")

    df["GR"]              = g["GR"].transform(lambda x: x.fillna(x.median()))
    # Forward fill last known TVT — critical for prediction zone
    df["TVT_last_known"]  = g["TVT_input"].transform(lambda x: x.ffill()) if "TVT_input" in df.columns else 0
    df["GR_roll_mean_10"] = g["GR"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df["GR_roll_mean_30"] = g["GR"].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df["GR_roll_std_10"]  = g["GR"].transform(lambda x: x.rolling(10, min_periods=1).std())
    df["GR_grad"]         = g["GR"].diff()
    df["MD_diff"]         = g["MD"].diff()
    df["Z_diff"]          = g["Z"].diff()
    df["Z_grad2"]         = g["Z"].diff().diff()
    df["TVT_known_grad"]  = g["TVT_last_known"].diff()
    df["TVT_known_grad2"] = g["TVT_last_known"].diff().diff()
    # Formation distance features — only available in train
    df["ANCC_filled"]     = g["ANCC"].transform(lambda x: x.fillna(x.median())) if "ANCC" in df.columns else 0
    df["dist_to_EGFDL"]   = df["Z"] - df["EGFDL"] if "EGFDL" in df.columns else 0
    df["dist_to_BUDA"]    = df["Z"] - df["BUDA"]  if "BUDA"  in df.columns else 0
    df["dist_to_ASTNU"]   = df["Z"] - df["ASTNU"] if "ASTNU" in df.columns else 0
    df["dist_to_ASTNL"]   = df["Z"] - df["ASTNL"] if "ASTNL" in df.columns else 0
    df["dist_to_EGFDU"]   = df["Z"] - df["EGFDU"] if "EGFDU" in df.columns else 0
    df["GR_x_TVT"]        = df["GR"] * df["TVT_last_known"]
    df["Z_x_TVT"]         = df["Z"]  * df["TVT_last_known"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

# Load and process training data
print("Loading train data...")
train_df = load_wells(TRAIN_DIR)
train_df = create_features(train_df)
print(f"Train rows: {len(train_df)} | Wells: {train_df['WELLNAME'].nunique()}")

# Use only rows with known TVT for training
train_known = train_df[train_df["TVT_last_known"] != 0].copy()
well_sample = train_known["WELLNAME"].unique()[:200]
train_cv    = train_known[train_known["WELLNAME"].isin(well_sample)].copy()

X_cv      = train_cv[FEATURES]
y_cv      = train_cv["TVT"]
groups_cv = train_cv["WELLNAME"]

print("\n====================")
print("CROSS VALIDATION")
print("====================")

gkf   = GroupKFold(n_splits=3)
model = HistGradientBoostingRegressor(
    learning_rate=0.03,
    max_depth=8,
    max_iter=500,
    min_samples_leaf=20,
    random_state=RANDOM_STATE
)

scores = []
for fold, (tr, te) in enumerate(gkf.split(X_cv, y_cv, groups_cv)):
    m = clone(model)
    m.fit(X_cv.iloc[tr], y_cv.iloc[tr])
    preds = m.predict(X_cv.iloc[te])
    rmse = np.sqrt(mean_squared_error(y_cv.iloc[te], preds))
    scores.append(rmse)
    print(f"  Fold {fold+1}: RMSE = {round(rmse, 4)}")

print(f"\nMean RMSE : {round(np.mean(scores), 4)}")
print(f"Std RMSE  : {round(np.std(scores), 4)}")

# Train final model on all available data
print("\nTraining final model on all data...")
model.fit(train_known[FEATURES], train_known["TVT"])

# Load and process test data
print("\nLoading test data...")
test_df = load_wells(TEST_DIR)
test_df = create_features(test_df)
print(f"Test rows: {len(test_df)} | Wells: {test_df['WELLNAME'].nunique()}")

test_df["tvt"]       = model.predict(test_df[FEATURES])
test_df["row_index"] = test_df.groupby("WELLNAME").cumcount()
test_df["id"]        = test_df["WELLNAME"] + "_" + test_df["row_index"].astype(str)

sample_sub = pd.read_csv("/kaggle/input/rogii-wellbore-geology-prediction/sample_submission.csv")
submission = sample_sub[["id"]].merge(test_df[["id", "tvt"]], on="id", how="left")
submission["tvt"] = submission["tvt"].fillna(submission["tvt"].median())

print("\n====================")
print("SUBMISSION")
print("====================")
print(f"Submission rows : {len(submission)}")
print(f"Nulls in tvt    : {submission['tvt'].isnull().sum()}")
print(submission.head())

# Must be saved here for Kaggle to detect it
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\nSaved → /kaggle/working/submission.csv")
