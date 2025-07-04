import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('E:\\Mall Customers\\Mall_Customers.csv')
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data.fillna(0, inplace=True)
features = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=5)
kmeans.fit(scaled_data)
clusters = kmeans.labels_

data['Cluster'] = clusters

plt.scatter(data['Spending Score (1-100)'], data['Annual Income (k$)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Annual Income (k$)')
plt.title('K-means Clustering (Spending Score (1-100) vs Annual Income)')
plt.show()

plt.scatter(data['Spending Score (1-100)'], data['Age'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Age')
plt.title('K-means Clustering (Spending Score (1-100) vs Age)')
plt.show()

plt.scatter(data['Spending Score (1-100)'], data['Gender'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Gender')
plt.title('K-means Clustering (Spending Score (1-100) vs Gender)')
plt.show()
