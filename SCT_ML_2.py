import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Mall_Customers.csv')

features = df[['Annual Income (k$)', 'Spending Score (1-100)'
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

centers = scaler.inverse_transform(kmeans.cluster_centers_)


type_labels = {
    0: 'Type 1: High Income, Low Spending',
    1: 'Type 2: Low Income, Low Spending',
    2: 'Type 3: Low Income, High Spending',
    3: 'Type 4: Average Income, Average Spending',
    4: 'Type 5: High Income, High Spending'
}


df['Customer Type'] = df['Cluster'].map(type_labels)
print("\n✅ Sample Labeled Customers:\n")
# print(df[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster', 'Customer Type']].head())
plt.figure(figsize=(12, 5))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Customer Type',
    data=df,
    palette='Set2',
    s=60
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c='black',
    s=200,
    alpha=0.6,
    marker='X',
    label='Centroids'
)
plt.title('Customer Segmentation Based on Income and Spending')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Customer Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
