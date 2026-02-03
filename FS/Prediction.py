import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from feature_selectors import ENetSelector, RFSelector, XGBSelector, LGBMSelector, LassoSelector, MISelector, BorutaSelector
from sklearn.base import clone

# ===========================
# Feature selection
# ===========================
fs_methods = {
    "ENet": ENetSelector(cv=10),
    "Lasso": LassoSelector(cv=10),
    "RF": RFSelector(n_estimators=100, max_depth=10),
    "XGB": XGBSelector(n_estimators=100),
    "LGBM": LGBMSelector(n_estimators=100),
    "MI": MISelector(k=500),
    "Boruta": BorutaSelector()
}

# ===========================
# Hyperparameter
# ===========================
models = {

    "RF": (
        RandomForestRegressor(random_state=42),
        {
            "model__n_estimators": [100],
            "model__max_depth": [3, 5, 10],
            "model__max_features": [0.1, 0.25, 0.5, 0.75, "sqrt", "log2", None],
        }
    ),

    "XGB": (
        XGBRegressor(objective="reg:squarederror", random_state=42),
        {
            "model__max_depth": [3, 5, 10],
        }
    ),

    "LGBM": (
        LGBMRegressor(random_state=42),
        {
            "model__max_depth": [3, 5, 10],
            "model__num_leaves": [31, 63],
        }
    ),

    "SVR_rbf": (
        SVR(kernel="rbf"),
        {
            "model__C": [0.001, 0.01, 0.1, 0.5, 1, 10, 50],
            "model__gamma": np.logspace(-5, 1, 7),
        }
    ),

    "SVR_poly": (
        SVR(kernel="poly"),
        {
            "model__C": [0.001, 0.01, 0.1, 0.5, 1, 10, 50],
            "model__degree": [2, 3, 4],
            "model__gamma": np.logspace(-5, 1, 7),
        }
    ),
}

# ===========================
# FS -> ML -> fold
# ===========================
data = pd.read_csv(r"D:\毕业\CAU毕业\发表文章\机器学习特征选择\投稿-ijms\scripts\transcriptome_data_sorted.csv",index_col=0)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
sample_ids = data.index
feature_names = data.columns[:-1]

results = []
N_REPEAT = 5
BASE_SEED = 42

for repeat in range(N_REPEAT):

    print(f"\n================ Repeat {repeat+1}/{N_REPEAT} ================")

    seed = BASE_SEED + repeat

    #CV
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=seed)

    for fs_name, fs_method in fs_methods.items():
        print(f"\n=== FS method: {fs_name} ===")

        for model_name, (base_model, param_grid) in models.items():
            print(f"\n--- Model: {model_name} ---")

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):

                print(f"\nRepeat {repeat+1}, Outer fold {fold_idx+1}")

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                steps = []

                if "SVR" in model_name:
                    steps.append(("scaler", StandardScaler()))

                # FS
                steps.append(("fs", clone(fs_method)))

                # model
                steps.append(("model", clone(base_model)))

                pipe = Pipeline(steps)

                # Inner CV
                inner_cv = KFold(
                    n_splits=9,
                    shuffle=True,
                    random_state=seed
                )

                grid = GridSearchCV(
                    pipe,
                    param_grid,
                    cv=inner_cv,
                    scoring="r2",
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_

                # ===============================
                # Export for RRBLUP.R
                # ===============================
                fs = best_model.named_steps["fs"]

                selected_features = fs.get_support(indices=True)

                feat_names = feature_names[selected_features]

                outdir = f"rrblup_data/{fs_name}_{model_name}/r{repeat + 1}_f{fold_idx + 1}"
                os.makedirs(outdir, exist_ok=True)

                X_fs_all = best_model.named_steps["fs"].transform(X)

                feat_idx = selected_features
                feat_names = feature_names[feat_idx]

                train_ids = sample_ids[train_idx]
                test_ids = sample_ids[test_idx]

                X_train_fs = X_fs_all[train_idx]
                X_test_fs = X_fs_all[test_idx]

                y_train_cv = y[train_idx]
                y_test_cv = y[test_idx]

                train_df = pd.DataFrame(
                    X_train_fs,
                    columns=feat_names
                )

                train_df.insert(0, "ID", train_ids)
                train_df["y"] = y_train_cv

                test_df = pd.DataFrame(
                    X_test_fs,
                    columns=feat_names
                )

                test_df.insert(0, "ID", test_ids)
                test_df["y"] = y_test_cv

                train_df.to_csv(f"{outdir}/train.csv", index=False)
                test_df.to_csv(f"{outdir}/test.csv", index=False)

                y_pred = best_model.predict(X_test)

                test_pcc = pearsonr(y_test, y_pred)[0]
                mean_inner_pcc = grid.best_score_

                selected_features = best_model.named_steps["fs"].get_support(indices=True)

                print(
                    f"Inner PCC = {mean_inner_pcc:.4f}, "
                    f"Test PCC = {test_pcc:.4f}, "
                    f"Features = {len(selected_features)}"
                )

                results.append({
                    "repeat": repeat + 1,
                    "fs": fs_name,
                    "model": model_name,
                    "fold": fold_idx + 1,
                    "mean_inner_pcc": mean_inner_pcc,
                    "test_pcc": test_pcc,
                    "best_params": grid.best_params_,
                    "selected_features": selected_features
                })
# ===========================
# Output
# ===========================
df_results = pd.DataFrame(results)

summary = df_results.groupby(["fs","model"]).agg(
    inner_mean = ("mean_inner_pcc", "mean"),
    inner_std  = ("mean_inner_pcc", "std"),
    test_mean  = ("test_pcc", "mean"),
    test_std   = ("test_pcc", "std"),
)

print("\nSummary:\n", summary)
