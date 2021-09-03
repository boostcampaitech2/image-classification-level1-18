import pandas as pd
from tqdm import tqdm

import config

from sklearn.metrics import f1_score

def main():
    my_df = pd.read_csv("efficientnet-b7-2021-09-02_131500-submission.csv")
    answer_df = pd.read_csv("answer.csv")

    print(answer_df.iloc[:5, -1].to_numpy())
    print(my_df.iloc[:5, -1].to_numpy())
    print(f1_score(answer_df.iloc[:, -1].to_numpy(), my_df.iloc[:, -1].to_numpy(), average='macro'))


if __name__ == "__main__":
    main()
