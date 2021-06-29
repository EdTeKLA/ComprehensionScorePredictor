from sklearn.model_selection import train_test_split
import pandas as pd

# grade = "3"
# df = pd.read_csv(f"../data/gr{grade}/gr{grade}_score.csv")

# train, test = train_test_split(df, test_size=0.1)

# train.to_csv(f"../data/gr{grade}/train.csv", index=False)
# test.to_csv(f"../data/gr{grade}/test.csv", index=False)

grade = "3"
df = pd.read_csv(f"../data/gr{grade}/gr{grade}_features.csv")

train, test = train_test_split(df, test_size=0.1)

train.to_csv(f"../data/gr{grade}/train_regression.csv", index=False)
test.to_csv(f"../data/gr{grade}/test_regression.csv", index=False)
