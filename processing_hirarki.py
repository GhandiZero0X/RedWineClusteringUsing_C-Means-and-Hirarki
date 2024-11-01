# Import pustaka yang dibutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage

# Load data
df = pd.read_csv('wine-clustering_Clean_MissingValue_Outlier.csv')
print("Data awal: \n", df.head())
print("\nData shape awal:", df.shape)

# Drop kolom 'class' jika ada
if 'class' in df.columns:
    df = df.drop(columns=['class'])
print("Data setelah menghapus kolom 'class': \n", df.head())
print("\nData shape setelah menghapus kolom 'class':", df.shape)

# Cek nilai null
df = df.dropna()
print("\nData setelah menghapus nilai null:", df.shape)

# Penghilangan outlier dengan batas yang lebih moderat menggunakan IQR 1.5
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print("\nData setelah menghapus outlier (IQR 1.5):", df.shape)

# Normalisasi data menggunakan Z-Score Standardization
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df)
print("\nData setelah normalisasi dengan Z-Score:\n", df_normalized[:5])

# Reduksi dimensi menggunakan PCA untuk mempertahankan 40% variance
pca = PCA(n_components=0.40)
df_pca = pca.fit_transform(df_normalized)
print("\nData setelah reduksi dimensi dengan PCA:\n", df_pca[:5])

# Menentukan nilai k optimal dengan menggunakan dendrogram (biasanya kriteria pemotongan manual)
# Linkage Matrix
Z = linkage(df_pca, method='ward')

# Visualisasi Dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Dendrogram untuk Clustering Hierarkis')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Menentukan jumlah cluster optimal secara otomatis dengan Silhouette Score
silhouette_scores = []
range_n_clusters = range(2, 16)  # Mencoba dari 2 hingga 15 cluster

for n_clusters in range_n_clusters:
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(df_pca)
    silhouette_avg = silhouette_score(df_pca, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Silhouette Score untuk {n_clusters} cluster: {silhouette_avg}")

# Visualisasi Silhouette Score untuk setiap jumlah cluster
plt.figure(figsize=(10, 5))
plt.plot(range_n_clusters, silhouette_scores, marker='o', color='blue')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score untuk Berbagai Jumlah Cluster')
plt.axhline(y=max(silhouette_scores), color='red', linestyle='--')
plt.show()

# Menentukan jumlah cluster optimal berdasarkan Silhouette Score tertinggi
optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
print(f"\nJumlah cluster optimal berdasarkan Silhouette Score: {optimal_n_clusters}")

# Melatih model Agglomerative Clustering dengan jumlah cluster optimal
model = AgglomerativeClustering(n_clusters=optimal_n_clusters)
labels = model.fit_predict(df_pca)

# Menghitung centroid untuk setiap cluster
centers = np.array([df_pca[labels == i].mean(axis=0) for i in range(optimal_n_clusters)])

# Visualisasi hasil clustering dalam 2D (proyeksi 2 fitur utama dari PCA)
plt.figure(figsize=(15, 7))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, s=50, cmap='viridis', label='Data Points')
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X', alpha=0.7, label='Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'Hasil Visualisasi Clustering Hierarkis dengan {optimal_n_clusters} Cluster')
plt.legend()
plt.show()

# Menghitung dan menampilkan Silhouette Score untuk clustering optimal
optimal_silhouette = silhouette_score(df_pca, labels)
print(f"\nSilhouette Score untuk clustering optimal: {optimal_silhouette}")

# Visualisasi Silhouette Score untuk setiap cluster
sample_silhouette_values = silhouette_samples(df_pca, labels)
y_lower = 10
plt.figure(figsize=(10, 6))
for i in range(optimal_n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / optimal_n_clusters)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

plt.axvline(x=optimal_silhouette, color="red", linestyle="--")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster label")
plt.title("Silhouette Plot for the Clusters")
plt.show()
