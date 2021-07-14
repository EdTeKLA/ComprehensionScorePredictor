from sklearn.model_selection import train_test_split
import pandas as pd

# grade = "3"
# df = pd.read_csv(f"../data/gr{grade}/gr{grade}_score.csv")

# train, test = train_test_split(df, test_size=0.1)

# train.to_csv(f"../data/gr{grade}/train.csv", index=False)
# test.to_csv(f"../data/gr{grade}/test.csv", index=False)

grade = "3"
# df = pd.read_csv(f"../data/gr{grade}/gr{grade}_score.csv")
df = pd.read_csv(f"../data/regression/gr3-gr4-gr5.csv")

train, test = train_test_split(df, test_size=0.2)

# train.to_csv(f"../data/gr{grade}/cg_train.csv", index=False)
# test.to_csv(f"../data/gr{grade}/cg_test.csv", index=False)

train.to_csv(f"../data/regression/train_gr3-gr4-gr5.csv", index=False)
test.to_csv(f"../data/regression/test_gr3-gr4-gr5.csv", index=False)
