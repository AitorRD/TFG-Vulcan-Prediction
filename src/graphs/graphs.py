import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
data = pd.read_csv("src/predict/results/submission.csv")

# Convert time_to_eruption from seconds to months (approximate conversion: 1 month ≈ 2,629,746 seconds)
data['time_to_eruption_months'] = data['time_to_eruption'] / 2629746


# Ensure the graphs directory exists
os.makedirs('src/graphs/plot', exist_ok=True)

# 1. Histogram of predicted time to eruption in months
plt.figure(figsize=(10, 6))
sns.histplot(data['time_to_eruption_months'], bins=20, kde=True)
plt.title('Histogram of Predicted Time to Eruption (Months)')
plt.xlabel('Time to Eruption (Months)')
plt.ylabel('Frequency')
plt.savefig('src/graphs/plot/histogram_time_to_eruption.png')
plt.close()