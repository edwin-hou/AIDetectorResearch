from sklearn.metrics import brier_score_loss
import pandas as pd

gpt_data = pd.read_csv("data.csv", usecols=[' zeroGPT Confidence'])
# print(gpt_data.loc[1][' zeroGPT Confidence'])

ground_truth = pd.read_csv("AI_Human.csv", nrows=gpt_data.shape[0], usecols=['generated'])
# print(ground_truth.loc[1])

gpt_array, truth_array = [], []
for _, row in gpt_data.iterrows():

    if float(row.iloc[0].replace('%', 'e-2'))>1:
        gpt_array.append(1.0)
    else:
        gpt_array.append(float(row.iloc[0].replace('%', 'e-2')))
for _, row in ground_truth.iterrows():
    truth_array.append(row.iloc[0])
# print(gpt_array)
brier_score = brier_score_loss(truth_array, gpt_array)
print(brier_score)