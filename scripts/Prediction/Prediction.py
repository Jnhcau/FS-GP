import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr


data = pd.read_csv("data.csv")  # 替换为你的文件
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


models = {
    "RF": (
        RandomForestRegressor(random_state=42),
        {
            "n_estimators": [100],
            "max_depth": [3, 5, 10],
            "max_features": [0.1, 0.25, 0.5, 0.75, "sqrt", "log2", None]
        }
    ),
    "XGB": (
        XGBRegressor(objective="reg:squarederror", random_state=42),
        {
            "n_estimators": [100],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.1, 0.5, 1],
            'max_features': [0.1, 0.5, 'sqrt', 'log2', None]
        }
    ),
    "LGBM": (
        LGBMRegressor(random_state=42),
        {
            "n_estimators": [300, 500],
            "num_leaves": [31, 63],
            "learning_rate": [0.05, 0.1]
        }
    ),
    "SVR_rbf": (
        SVR(kernel="rbf"),
        {
            'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 50],
            'gamma': np.logspace(-5, 1, 7)
        }
    ),
    "SVR_poly": (
        SVR(kernel="poly"),
        {
            'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 50],
            'degree': [2, 3, 4],
            'gamma': np.logspace(-5, 1, 7)
        }
    )
}

outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
results = {name: [] for name in models.keys()}

for fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(X)):
    print(f"\n=== Outer fold {fold_idx + 1} ===")

    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y[train_val_idx], y[test_idx]


    inner_cv = KFold(n_splits=9, shuffle=True, random_state=42)

    for name, (model, param_grid) in models.items():
        print(f"\nTraining model: {name}")

        # GridSearchCV 进行内层 CV
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="r2",
            n_jobs=-1
        )
        grid.fit(X_train_val, y_train_val)


        best_model = grid.best_estimator_
        inner_val_pccs = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val):
            X_inner_train, X_inner_val = X_train_val[inner_train_idx], X_train_val[inner_val_idx]
            y_inner_val = y_train_val[inner_val_idx]
            y_inner_pred = best_model.fit(X_inner_train, y_train_val[inner_train_idx]).predict(X_inner_val)
            pcc = pearsonr(y_inner_val, y_inner_pred)[0]
            inner_val_pccs.append(pcc)
        mean_inner_pcc = np.mean(inner_val_pccs)


        y_test_pred = best_model.predict(X_test)
        test_pcc = pearsonr(y_test, y_test_pred)[0]

        print(f"{name} Mean inner PCC: {mean_inner_pcc:.4f}, Test PCC: {test_pcc:.4f}")

        results[name].append({
            "fold": fold_idx + 1,
            "mean_inner_pcc": mean_inner_pcc,
            "test_pcc": test_pcc,
            "best_params": grid.best_params_
        })


for name, folds in results.items():
    inner_pccs = [f["mean_inner_pcc"] for f in folds]
    test_pccs = [f["test_pcc"] for f in folds]
    print(f"\n{name}:")
    print(f"  Outer folds Mean inner PCC = {np.mean(inner_pccs):.4f} ± {np.std(inner_pccs):.4f}")
    print(f"  Outer folds Test PCC = {np.mean(test_pccs):.4f} ± {np.std(test_pccs):.4f}")
