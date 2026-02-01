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

# ===========================
# 数据
# ===========================
data = pd.read_csv("data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# ===========================
# 特征选择列表
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
# ML 模型 + 超参数
# ===========================
models = {
    "RF": (RandomForestRegressor(random_state=42),
           {"n_estimators": [100],
            "max_depth": [3, 5, 10],
            "max_features": [0.1, 0.25, 0.5, 0.75, "sqrt", "log2", None]}),
    "XGB": (XGBRegressor(objective="reg:squarederror", random_state=42),
            {"max_depth": [3, 5, 10]}),
    "LGBM": (LGBMRegressor(random_state=42),
             {"max_depth": [3, 5, 10], "num_leaves": [31, 63]}),
    "SVR_rbf": (SVR(kernel="rbf"),
                {"C": [0.001,0.01,0.1,0.5,1,10,50],
                 "gamma": np.logspace(-5,1,7)}),
    "SVR_poly": (SVR(kernel="poly"),
                 {"C": [0.001,0.01,0.1,0.5,1,10,50],
                  "degree": [2,3,4],
                  "gamma": np.logspace(-5,1,7)})
}

# ===========================
# 外层 CV
# ===========================
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

# 保存结果
results = []

# ===========================
# 主循环: FS -> ML -> fold
# ===========================
for fs_name, fs_method in fs_methods.items():
    print(f"\n=== FS method: {fs_name} ===")

    for model_name, (base_model, param_grid) in models.items():
        print(f"\n--- Model: {model_name} ---")

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
            print(f"\nOuter fold {fold_idx+1}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            steps = []

            # SVR 需要标准化
            if "SVR" in model_name:
                steps.append(("scaler", StandardScaler()))

            # FS
            steps.append(("fs", fs_method))

            # 模型
            steps.append(("model", base_model))

            pipe = Pipeline(steps)

            # 内层 CV 调参
            inner_cv = KFold(n_splits=9, shuffle=True, random_state=42)
            grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="r2", n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            test_pcc = pearsonr(y_test, y_pred)[0]
            mean_inner_pcc = grid.best_score_

            # 获取选中特征索引
            selected_features = best_model.named_steps["fs"].get_support(indices=True)

            print(f"Inner PCC (GridSearchCV) = {mean_inner_pcc:.4f}, Test PCC = {test_pcc:.4f}, Selected features = {len(selected_features)}")

            fold_results.append({
                "fs": fs_name,
                "model": model_name,
                "fold": fold_idx+1,
                "mean_inner_pcc": mean_inner_pcc,
                "test_pcc": test_pcc,
                "best_params": grid.best_params_,
                "selected_features": selected_features
            })

        results.extend(fold_results)

# ===========================
# 汇总输出
# ===========================
import pandas as pd
df_results = pd.DataFrame(results)
summary = df_results.groupby(["fs","model"]).agg({
    "mean_inner_pcc": ["mean","std"],
    "test_pcc": ["mean","std"]
})
print("\nSummary:\n", summary)
