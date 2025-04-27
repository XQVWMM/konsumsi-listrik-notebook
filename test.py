import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


data_2021 = pd.read_csv('/Users/Jeff/Documents/ML/DATASET/2021/Hasil_2021.csv')
data_2022 = pd.read_csv('/Users/Jeff/Documents/ML/DATASET/2022/Hasil_2022.csv')
data_2023 = pd.read_csv('/Users/Jeff/Documents/ML/DATASET/2023/Hasil_2023.csv')
data_JP = pd.read_csv('/Users/Jeff/Documents/ML/DATASET/Jumlah penduduk/Hasil_JP.csv')
data_KP = pd.read_csv('/Users/Jeff/Documents/ML/DATASET/Kapasitas Pembangkit/Hasil_KP.csv')
data_final = pd.read_csv('Hasil_Gabungan.csv')
features = ['Residential_2021', 'Business_2021', 'Industrial_2021', 'Social_2021', 'Gov_Office_2021', 'Pub_Street_2021', 'Total_2021', 'JP_2021', 'KP_2021']

data_T2021 = data_final[features]
robust_scaler = RobustScaler()
data_scaled_2021 = robust_scaler.fit_transform(data_T2021)

kmeans_robust = KMeans(n_clusters=3, random_state=42)
data_T2021['Cluster_Robust'] = kmeans_robust.fit_predict(data_scaled_2021)

centroids_robust = kmeans_robust.cluster_centers_
print("Centroid dengan RobustScaler:", centroids_robust)
data_final['Cluster_Robust'] = kmeans_robust.fit_predict(data_scaled_2021)
provinsi_per_cluster = data_final[['Province', 'Cluster_Robust']].sort_values(by='Cluster_Robust')

for cluster in range(3):  
    print(f"Provinsi dalam Cluster {cluster}:")
    print(provinsi_per_cluster[provinsi_per_cluster['Cluster_Robust'] == cluster]['Province'].tolist())
    print("\n")
features = ['Residential_2022', 'Business_2022', 'Industrial_2022', 'Social_2022', 'Gov_Office_2022', 'Pub_Street_2022', 'Total_2022', 'JP_2022', 'KP_2022']

data_T2022 = data_final[features]

robust_scaler2 = RobustScaler()
data_scaled_2022 = robust_scaler2.fit_transform(data_T2022)

kmeans_robust2 = KMeans(n_clusters=3, random_state=42)
data_T2022['Cluster_Robust'] = kmeans_robust2.fit_predict(data_scaled_2022)

centroids_robust2 = kmeans_robust2.cluster_centers_
print("Centroid dengan RobustScaler:", centroids_robust2)
data_final['Cluster_Robust'] = kmeans_robust2.fit_predict(data_scaled_2022)
provinsi_per_cluster2 = data_final[['Province', 'Cluster_Robust']].sort_values(by='Cluster_Robust')

for cluster in range(3):  
    print(f"Provinsi dalam Cluster {cluster}:")
    print(provinsi_per_cluster2[provinsi_per_cluster2['Cluster_Robust'] == cluster]['Province'].tolist())
    print("\n")

features = ['Residential', 'Business', 'Industrial', 'Social', 'Gov_Office', 'Pub_Street', 'Total', 'JP_2023', 'KP_2023']

data_T2023 = data_final[features]

robust_scaler3 = RobustScaler()
data_scaled_2023 = robust_scaler3.fit_transform(data_T2023)

kmeans_robust3 = KMeans(n_clusters=3, random_state=42)
data_T2023['Cluster_Robust'] = kmeans_robust3.fit_predict(data_scaled_2023)

centroids_robust3 = kmeans_robust3.cluster_centers_
print("Centroid dengan RobustScaler:", centroids_robust3)
data_final['Cluster_Robust'] = kmeans_robust3.fit_predict(data_scaled_2023)
provinsi_per_cluster3 = data_final[['Province', 'Cluster_Robust']].sort_values(by='Cluster_Robust')

for cluster in range(3):  
    print(f"Provinsi dalam Cluster {cluster}:")
    print(provinsi_per_cluster3[provinsi_per_cluster3['Cluster_Robust'] == cluster]['Province'].tolist())
    print("\n")