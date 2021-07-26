import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import itertools
import numpy as np
from sklearn import preprocessing

feature_path = "../SoR_Alberta.Shared.Data.and.Codebook.xlsx"#"../data/gr3/gr3_features.xlsx"
feature_names = ['G3.PPVT.Vocab.raw',
                 'G3.Elision.PA.raw',
                 'G3.Syn.GramCorrect.raw',
                 'G4.TOWRE.SWE.raw',
                 'G4.TOWRE.PDE.raw',
                 'G4.WordID.raw',
                 'G3.OL.Spell.Total',
                 'G3.OL.OrthoChoice.1.2.Total',
                 'G3.DigitSpan.raw',
                 'G3.Gates.RC.raw',
                 'G4.Gates.RC.raw',]
target_path = "../data/regression/gr3-gr4.csv"
score_name = "G3.Gates.RC.raw"

df = pd.read_excel(feature_path)

data = {}
for name in feature_names:
    data[name] = []
for i in df.index:
    unavailable = False
    for col in feature_names:
        if df[col][i] < 0:
            unavailable = True
            break
    
    if unavailable:
        continue
    for name in feature_names:
        data[name].append(df[name][i])

# normalize each feature with max norm
for name in feature_names[0:-1]:
    new_data = np.asarray(data[name]).reshape(-1, 1)
    data[name] = preprocessing.normalize(new_data, norm='max',axis=0).reshape(-1,1).squeeze()

target_df = pd.DataFrame(data, columns = feature_names)
target_df.to_csv(target_path, index=False)