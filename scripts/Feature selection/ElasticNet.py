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
    # 2. Elastic Net feature selection
    # ==========================
    enet = ElasticNetCV(
        cv=5,
        random_state=42,
        max_iter=10000,
        n_jobs=-1
    )

    enet.fit(X, y)

    coef = enet.coef_

    enet_genes = feature_names[np.abs(coef) > 0]

    enet_df = pd.DataFrame({
        "Feature": enet_genes,
        "Coefficient": coef[np.abs(coef) > 0]
    })

    enet_df.to_csv(
        f"TRA_{trait}_ElasticNet_features.csv",
        index=False
    )

    print(f"Elastic Net selected {len(enet_genes)} features")