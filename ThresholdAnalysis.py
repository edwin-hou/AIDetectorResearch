from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv("data.csv")

thresholds = [i/100 for i in range(101)]
results = []
y_true = data['truth'] / 100  # Convert truth to 0 and 1
y_scores = data['zeroGPT'] / 100  # Normalize zeroGPT scores

for threshold in thresholds:
    predictions = (y_scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average='binary')
    results.append((threshold, precision, recall, f1))

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=['Threshold', 'Precision', 'Recall', 'F1'])

# Plot the metrics
plt.figure(figsize=(12, 6))
plt.plot(results_df['Threshold'], results_df['Precision'], label='Precision')
plt.plot(results_df['Threshold'], results_df['Recall'], label='Recall')
plt.plot(results_df['Threshold'], results_df['F1'], label='F1 Score')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Sensitivity Analysis')
plt.legend()
plt.show()
