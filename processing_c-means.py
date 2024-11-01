# Import pustaka yang dibutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from fcmeans import FCM

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

# Reduksi dimensi menggunakan PCA untuk mempertahankan 85% variance
pca = PCA(n_components=0.40)
df_pca = pca.fit_transform(df_normalized)
print("\nData setelah reduksi dimensi dengan PCA:\n", df_pca[:5])

# Menentukan nilai k optimal menggunakan Elbow Method pada C-Means dengan eksperimen lebih luas
range_n_clusters = range(1, 16)
inertia = []

for n_clusters in range_n_clusters:
    model = FCM(n_clusters=n_clusters, m=1.8)
    model.fit(df_pca)
    u = model.u
    centers = model.centers
    fuzzy_inertia = np.sum((u**2) * np.linalg.norm(df_pca[:, np.newaxis] - centers, axis=2)**2)
    inertia.append(fuzzy_inertia)

# Menentukan jumlah cluster optimal dengan Elbow Method
inertia_diff = np.diff(inertia)
inertia_diff_ratio = inertia_diff[:-1] / inertia_diff[1:]
optimal_n_clusters = np.argmax(inertia_diff_ratio) + 2
print(f"\nJumlah cluster optimal berdasarkan Elbow Method: {optimal_n_clusters}")

# Visualisasi Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(range_n_clusters, inertia, marker='o', color='orange')
plt.xlabel('Number of Clusters')
plt.ylabel('Fuzzy Inertia')
plt.title('Elbow Method for Optimal Number of Clusters in C-Means')
plt.show()

# Melatih model FCM dengan jumlah cluster optimal dan visualisasi hasil clustering
model = FCM(n_clusters=optimal_n_clusters, m=1.8)
model.fit(df_pca)
labels = model.predict(df_pca)
centers = model.centers

# Visualisasi hasil clustering dalam 2D (proyeksi 2 fitur utama dari PCA)
plt.figure(figsize=(15, 7))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'FCM Clustering Visualization with {optimal_n_clusters} Clusters')
plt.show()

# Menghitung dan menampilkan Silhouette Score untuk clustering optimal
optimal_silhouette = silhouette_score(df_pca, labels)
print(f"Silhouette Score untuk clustering optimal: {optimal_silhouette}")

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
