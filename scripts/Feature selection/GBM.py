import pandas as pd
import xgboost as xgb
import lightgbm as lgb
snp_matrix = pd.read_csv("transcriptome.csv", index_col=0)
phenotypes = pd.read_csv("pheno.csv", index_col=0)

phenotype_columns = ['FT']
for phenotype_column in phenotype_columns:
    print(f"分析表型: {phenotype_column}")

    y = phenotypes[phenotype_column].values

    X = snp_matrix.values

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        objective='reg:squarederror',
        random_state=41
    )

    xgb_model.fit(X, y)

    xgb_importance = xgb_model.feature_importances_

    xgb_importance_df = pd.DataFrame({
        'SNP': snp_matrix.columns,
        'Importance': xgb_importance
    })

    xgb_importance_df = xgb_importance_df.sort_values(by='Importance', ascending=False)

    xgb_output_file = f"TRA_{phenotype_column}_xgb_selected_genes_ranked.csv"
    xgb_importance_df.to_csv(xgb_output_file, index=False)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,  # 迭代次数
        objective='regression',  # 回归任务
        random_state=41
    )

    lgb_model.fit(X, y)

    lgb_importance = lgb_model.feature_importances_
数
    lgb_importance_df = pd.DataFrame({
        'SNP': snp_matrix.columns,
        'Importance': lgb_importance
    })

    lgb_importance_df = lgb_importance_df.sort_values(by='Importance', ascending=False)

    lgb_output_file = f"/public/home/04034/ddyy/machine_learning/TRA_{phenotype_column}_lgb_selected_genes_ranked.csv"
    lgb_importance_df.to_csv(lgb_output_file, index=False)

