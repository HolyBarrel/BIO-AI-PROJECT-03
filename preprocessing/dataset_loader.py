import pandas as pd
#import cupy as cp

# Read the TSV file
bc_df = pd.read_csv('data/breast_cancer_wisconsin_original.tsv', sep='\t')

wq_df = pd.read_csv('data\wine_quality_combined.tsv', sep='\t')

# TODO: Add later when processing on GPU
#data_cp = cp.array(df.values)

# Display the first few rows to verify
print(bc_df.head())

print(wq_df.head())

