import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
data = pd.read_csv("src/predict/results/results_gboost_crossval_manualfeatures.csv")

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

# 2. Scatter Plot of predicted time to eruption vs. volcano ID
plt.figure(figsize=(14, 8))
sns.scatterplot(x='volcan_id', y='time_to_eruption_months', data=data)
plt.title('Scatter Plot of Predicted Time to Eruption vs. Volcano ID')
plt.xlabel('Volcano ID')
plt.ylabel('Time to Eruption (Months)')
plt.xticks(rotation=90)
plt.savefig('src/graphs/plot/scatter_volcan_id_time_to_eruption.png')
plt.close()

# 3. Box Plot of predicted time to eruption in months
plt.figure(figsize=(10, 6))
sns.boxplot(y='time_to_eruption_months', data=data)
plt.title('Box Plot of Predicted Time to Eruption (Months)')
plt.ylabel('Time to Eruption (Months)')
plt.savefig('src/graphs/plot/boxplot_time_to_eruption.png')
plt.close()

# 4. Bar Plot of predicted time to eruption by volcano
plt.figure(figsize=(14, 8))
sns.barplot(x='volcan_id', y='time_to_eruption_months', data=data)
plt.title('Bar Plot of Predicted Time to Eruption by Volcano')
plt.xlabel('Volcano ID')
plt.ylabel('Time to Eruption (Months)')
plt.xticks(rotation=90)
plt.savefig('src/graphs/plot/barplot_volcan_id_time_to_eruption.png')
plt.close()

# 5. Line Plot of predictions sorted by time to eruption
plt.figure(figsize=(10, 6))
sorted_data = data.sort_values('time_to_eruption_months')
sns.lineplot(x=range(len(sorted_data)), y='time_to_eruption_months', data=sorted_data)
plt.title('Line Plot of Predictions Sorted by Time to Eruption')
plt.xlabel('Sorted Volcanoes')
plt.ylabel('Time to Eruption (Months)')
plt.savefig('src/graphs/plot/lineplot_sorted_time_to_eruption.png')
plt.close()