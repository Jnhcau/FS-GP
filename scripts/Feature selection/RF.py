import pandas as pd
from sklearn.ensemble import RandomForestRegressor

snp_matrix = pd.read_csv("transcriptome.csv", index_col=0)
phenotypes = pd.read_csv("pheno.csv", index_col=0)

phenotype_columns = ['FT']

for phenotype_column in phenotype_columns:
    print(f"分析表型: {phenotype_column}")

    y = phenotypes[phenotype_column].values

    X = snp_matrix.values

    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    rf.fit(X, y)

    importance = rf.feature_importances_

    importance_df = pd.DataFrame({
        'SNP': snp_matrix.columns,
        'Importance': importance
    })


    selected_genes = importance_df[importance_df['Importance'] > 0]['SNP']

    # 保存选择的基因到文件
    output_file = f"TRA_{phenotype_column}_rf_selected_genes.txt"
    with open(output_file, 'w') as f:
        for gene in selected_genes:
            f.write(gene + '\n')

