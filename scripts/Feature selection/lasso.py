import pandas as pd
from sklearn.linear_model import LassoCV

# 读取 SNP 数据和表型数据
snp_matrix = pd.read_csv("transcriptome.csv", index_col=0)
phenotypes = pd.read_csv("pheno.csv", index_col=0)


phenotype_columns = ['FT']

for phenotype_column in phenotype_columns:
    print(f"分析表型: {phenotype_column}")

    y = phenotypes[phenotype_column].values

    X = snp_matrix.values

    lasso = LassoCV(cv=5, random_state=41, max_iter=5000)
    lasso.fit(X, y)

    selected_genes = snp_matrix.columns[lasso.coef_ != 0]

    output_file = f"TRA_{phenotype_column}_lasso_selected_genes.txt"
    with open(output_file, 'w') as f:
        for gene in selected_genes:
            f.write(gene + '\n')

