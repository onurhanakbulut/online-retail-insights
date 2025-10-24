import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


rfm_summary = pd.DataFrame({
    'Segment' : ['Champions', 'Loyal', 'Potential Loyalist', 'Regular', 'At Risk', 'Hibernating'],
    'Avg_Recency': [11.4, 38.9, 14.5, 75.3, 148.8, 274.8],
    'Avg_Frequency': [10.7, 5.4, 2.3, 1.3, 3.4, 1.1],
    'Avg_Monetary': [5827, 2121, 1283, 523, 1376, 226]
    })


sns.barplot(data=rfm_summary, x='Segment', y='Avg_Monetary', palette='viridis')
plt.xticks(rotation=45)
plt.title('Average Monetary by Segment')
plt.show()


sns.heatmap(rfm_summary.set_index('Segment')[['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']],
            annot=True, cmap='YlGnBu')
plt.title('RFM Segment Profile Heatmap')
plt.show()
