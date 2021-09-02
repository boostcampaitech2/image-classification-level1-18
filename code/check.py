import pandas as pd
from tqdm import tqdm


from sklearn.metrics import f1_score
#submission_fcb_b4.csv
#submissione_f_b4_50.csv
my_df = pd.read_csv('/opt/ml/input/data/eval/submission_b4_ense.csv')
answer_df = pd.read_csv("/opt/ml/code/bnbbnb.csv")

print(answer_df.iloc[:5, -1].to_numpy())
print(my_df.iloc[:5, -1].to_numpy())
print(f1_score(answer_df.iloc[:, -1].to_numpy(), my_df.iloc[:, -1].to_numpy(), average='macro'))