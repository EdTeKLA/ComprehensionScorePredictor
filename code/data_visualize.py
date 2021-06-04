import pandas as pd
import numpy as np
import os

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
    df = pd.read_csv('../data/ReadComp_clean.csv')
    gr3 = []
    gr4 = []
    gr5 = []
    for i in range(df.shape[0]):
        if df["Gr3"][i] != -99:
            gr3.append(df["Gr3"][i])
        if df["Gr4"][i] != -99:
            gr4.append(df["Gr4"][i])
        if df["Gr5"][i] != -99:
            gr5.append(df["Gr5"][i])

    print(f"Number of students who took Gr3 test: {len(gr3)}\n"
          f"Number of students who took Gr4 test: {len(gr4)}\n"
          f"Number of students who took Gr5 test: {len(gr5)}")
    
    print(np.mean(gr3),np.std(gr3))

if __name__ == '__main__':
    main()