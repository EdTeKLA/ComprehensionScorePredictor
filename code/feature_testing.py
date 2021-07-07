import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import itertools
import numpy as np

feature_path = "../data/gr3/gr3_features.xlsx"

df = pd.read_excel(feature_path)
print(df.columns)

# testing for correlation between the vairable and the target to check for
# significance
for col in df.columns[1:-1]:
    test_var = df[col]
    score = df["G3.Gates.RC.raw"]

    print(col)
    corr, _ = spearmanr(test_var, score)

    print("Spearman's correlation: %.3f" % corr)

    corr, _ = pearsonr(test_var, score)
    print("Pearson's correlation: %.3f" % corr)

# testing for correlated variables to avoid redundancy
for a,b in itertools.combinations(df.columns[1:-1],2):
    corr, _ = pearsonr(df[a], df[b])
    if corr > 0.5:
        print(a,b)
        print(corr)