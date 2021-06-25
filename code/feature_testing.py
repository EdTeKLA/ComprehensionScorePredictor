import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr

feature_path = "../data/gr3/gr3_features.xlsx"

df = pd.read_excel(feature_path)
test_var = df["G3.OL.OrthoChoice.1.2.Total"]
score = df["G3.Gates.RC.nce"]

corr, _ = spearmanr(test_var, score)

print("Spearman's correlation: %.3f" % corr)

corr, _ = pearsonr(test_var, score)
print("Pearson's correlation: %.3f" % corr)