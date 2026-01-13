import pandas as pd
from sklearn.feature_selection import mutual_info_regression

snp_matrix = pd.read_csv("transcriptome.csv", index_col=0)
phenotypes = pd.read_csv("pheno.csv", index_col=0)

phenotype_columns = ['FT']

for phenotype_column in phenotype_columns:
    print(f"分析表型: {phenotype_column}")

    y = phenotypes[phenotype_column].values

    X = snp_matrix.values

    mi_scores = mutual_info_regression(X, y)

    importance_df = pd.DataFrame({
        'SNP': snp_matrix.columns,
        'Importance': mi_scores
    })

    selected_genes = importance_df[importance_df['Importance'] > 0]['SNP']

    output_file = f"TRA_{phenotype_column}_mutualinfo_selected_genes.txt"
    with open(output_file, 'w') as f:
        for gene in selected_genes:
            f.write(gene + '\n')

