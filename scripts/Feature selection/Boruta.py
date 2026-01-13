import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from sklearn.linear_model import ElasticNetCV
# --------------------------
# Load data
# --------------------------
X_df = pd.read_csv("transcriptome.csv", index_col=0)
y_df = pd.read_csv("pheno.csv", index_col=0)

phenotype_columns = ["FT"]

X = X_df.values
feature_names = X_df.columns

# --------------------------
# Loop over phenotypes
# --------------------------
for trait in phenotype_columns:
    print(f"Processing trait: {trait}")
    y = y_df[trait].values

    # ==========================
    # 1. Boruta feature selection
    # ==========================
    rf = RandomForestRegressor(
        n_jobs=-1,
        random_state=42
    )

    boruta = BorutaPy(
        rf,
        n_estimators="auto",
        verbose=2,
        random_state=42
    )

    boruta.fit(X, y)

    boruta_genes = feature_names[boruta.support_]

    with open(f"TRA_{trait}_Boruta_features.txt", "w") as f:
        for g in boruta_genes:
            f.write(g + "\n")

    print(f"Boruta selected {len(boruta_genes)} features")