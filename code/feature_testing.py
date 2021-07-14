import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import itertools
import numpy as np

feature_path = "../SoR_Alberta.Shared.Data.and.Codebook.xlsx"#"../data/gr3/gr3_features.xlsx"
score_name = "G3.Gates.RC.raw"

feature_names = ['G3.PPVT.Vocab.raw',
                 'G3.Elision.PA.raw',
                 'G3.Syn.GramCorrect.raw',
                 'G3.TOWRE.SWE.raw',
                 'G3.TOWRE.PDE.raw',
                 'G3.WordID.raw',
                 'G3.OL.Spell.Total',
                 'G3.OL.OrthoChoice.1.2.Total',
                 'G3.DigitSpan.raw',
                 'G3.Gates.RC.raw',
                 'G4.Gates.RC.raw',
                 'G5.Gates.RC.raw']

df = pd.read_excel(feature_path)
print(df.columns)

# testing for correlation between the vairable and the target to check for
# significance
# for col in df.columns[1:]:
#     if "RC.Gates" in col:
#         continue

for col in feature_names:
    test_var = []
    score = []
    for i in df.index:
        if df[col][i] >= 0:
            test_var.append(df[col][i])
            score.append(df[score_name][i])
    print(col)
    corr, _ = spearmanr(test_var, score)

    print("Spearman's correlation: %.3f" % corr)

    corr, _ = pearsonr(test_var, score)
    print("Pearson's correlation: %.3f" % corr)

# testing for correlated variables to avoid redundancy
# for a,b in itertools.combinations(df.columns[1:-1],2):
#     corr, _ = pearsonr(df[a], df[b])
#     if corr > 0.5:
#         print(a,b)
#         print(corr)