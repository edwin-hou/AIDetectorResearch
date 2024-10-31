from sklearn.metrics import brier_score_loss
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

gpt_data = pd.read_csv("data.csv", usecols=[' zeroGPT Confidence'])
# print(gpt_data.loc[1][' zeroGPT Confidence'])

ground_truth = pd.read_csv("AI_Human.csv", nrows=gpt_data.shape[0], usecols=['generated'])
# print(ground_truth.loc[1])

gpt_array, truth_array, difference_array = [], [], []
for _, row in gpt_data.iterrows():

    if float(row.iloc[0].replace('%', 'e-2')) > 1:
        gpt_array.append(1.0)
    else:
        gpt_array.append(float(row.iloc[0].replace('%', 'e-2')))
for _, row in ground_truth.iterrows():
    truth_array.append(row.iloc[0])

for i in range(len(gpt_data)):
    difference_array.append((truth_array[i] - gpt_array[i]) * 100)

values, bins, bars = plt.hist(difference_array, bins=20)
plt.ylabel('Frequency')
plt.xlabel('% Difference')
plt.title("Difference between Ground Truth and ZeroGPT's prediction")
plt.xticks(np.arange(-100, 110, 10))
plt.bar_label(bars, fontsize=10, color='navy')

plt.show()
