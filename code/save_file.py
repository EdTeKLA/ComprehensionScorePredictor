from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("../data/gr3_test_to_score.csv")

train, test = train_test_split(df, test_size=0.2)

train.to_csv("../data/train.csv", index=False)
test.to_csv("../data/test.csv", index=False)