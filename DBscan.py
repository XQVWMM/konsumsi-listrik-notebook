import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.manifold import TSNE
import leafmap as lf
import geopandas as gpd
data = pd.read_csv('../DATASET/Hasil_Gabungan.csv')

featuress = [
    'Residential_2021', 'Industrial_2021', 'Business_2021', 'Social_2021', 'Gov_Office_2021', 
    'Pub_Street_2021', 'Total_2021', 'Residential_2022', 'Industrial_2022', 'Business_2022', 
    'Social_2022', 'Gov_Office_2022', 'Pub_Street_2022', 'Total_2022', 'Residential', 
    'Industrial', 'Business', 'Social', 'Gov_Office', 'Pub_Street', 'Total', 
    'JP_2021', 'JP_2022', 'JP_2023', 'KP_2021', 'KP_2022', 'KP_2023'
]

data_T = data[featuress]
scaler = RobustScaler()
scaled_features = scaler.fit_transform(data_T)


dbscan = DBSCAN(eps=9, min_samples=16)
clusters = dbscan.fit_predict(scaled_features)

data['Cluster'] = clusters

for cluster_id in sorted(set(clusters)):
    provinsi = data[data['Cluster'] == cluster_id]['Province'].tolist()
    if cluster_id == -1:
        print("\nOutlier:")
    else:
        print(f"\nCluster {cluster_id}:")
    print(provinsi)

valid_mask = clusters != -1
if len(set(clusters[valid_mask])) > 1 and valid_mask.sum() > 1:
    sil_score_no_outliers = silhouette_score(
        scaled_features[valid_mask], clusters[valid_mask]
    )
    print(f"\nSilhouette Score (tanpa outliers): {sil_score_no_outliers:}")
else:
    print("\nSilhouette Score (tanpa outliers) tidak dapat dihitung.")

if len(set(clusters)) > 1:
    sil_score_with_outliers = silhouette_score(scaled_features, clusters)
    print(f"Silhouette Score (dengan outliers): {sil_score_with_outliers:}")
else:
    print("Silhouette Score (dengan outliers) tidak dapat dihitung.")


geo_data = gpd.read_file('../DATASET/id.json')

geo_data['name'] = geo_data['name'].replace('Jakarta Raya', 'DKI Jakarta')
geo_data['name'] = geo_data['name'].replace('Kepulauan Riau', 'Kep. Riau')
geo_data['name'] = geo_data['name'].replace('Yogyakarta', 'DI Yogyakarta')
geo_data['name'] = geo_data['name'].replace('Bangka-Belitung', 'Kep. Bangka Belitung')
geo_data['name'] = geo_data['name'].replace('North Kalimantan', 'Kalimantan Utara')

merged_data = geo_data.merge(data, left_on='name', right_on='Province')

m = lf.Map(center=[-6.1751, 106.8650], zoom=5)

cluster_colors = {
    1: 'blue',  
    0: 'green', 
    -1: 'red'    
}

for cluster in merged_data['Cluster'].unique():
    cluster_data = merged_data[merged_data['Cluster'] == cluster]

    cluster_data['color'] = cluster_data['Cluster'].map(cluster_colors)

    m.add_gdf(cluster_data, layer_name=f"Cluster {cluster}", color_col="color")

m.add_layer_control()

m