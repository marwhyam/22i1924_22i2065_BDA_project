import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from multiprocessing import Pool, cpu_count
import librosa

# Configuration
CONFIG = {
    'fma_large_path': "/media/maryam/Seagate Expansion Drive/dataframe/fma_large (2)",
    'fma_metadata_path': "/home/maryam/Downloads/fma_metadata",
    'mongo_host': 'localhost',
    'mongo_port': 27017,
    'mongo_db': 'feature_extract',
    'mongo_collection': 'extraction',
    'num_clusters': 10,
    'batch_size': 100,  # Processing files in batches of 100
    'max_files': 106000 # Maximum number of files to process
}

def connect_to_mongodb():
    try:
        client = MongoClient(CONFIG['mongo_host'], CONFIG['mongo_port'], serverSelectionTimeoutMS=5000)
        print("Connected to MongoDB successfully.")
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def convert_numpy_to_python(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, dict):
        return {k: convert_numpy_to_python(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python(v) for v in data]
    else:
        return data

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=30)
        if len(y) == 0:
            print(f"Warning: Empty audio file detected at {file_path}.")
            return None
        features = {
            'mfcc': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0),
            'file_path': file_path
        }
        print(f"Features extracted for file: {file_path}.")
        return features
    except Exception as e:
        print(f"Error extracting features for {file_path}: {e}")
        return None

def process_batch(file_paths):
    with Pool(processes=cpu_count()) as pool:
        features = pool.map(extract_features, file_paths)
    return [f for f in features if f is not None]

def main():
    client = connect_to_mongodb()
    if not client:
        return

    db = client[CONFIG['mongo_db']]
    collection = db[CONFIG['mongo_collection']]

    tracks = pd.read_csv(os.path.join(CONFIG['fma_metadata_path'], 'tracks.csv'), index_col=0, header=[0, 1])
    track_ids = set(tracks.index)
    all_features = []
    folders = [os.path.join(CONFIG['fma_large_path'], folder) for folder in os.listdir(CONFIG['fma_large_path'])]
    processed_files = 0

    for folder in folders:
        if processed_files >= CONFIG['max_files']:
            break
        file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if int(os.path.splitext(f)[0]) in track_ids]
        for i in range(0, len(file_paths), CONFIG['batch_size']):
            batch_paths = file_paths[i:i + CONFIG['batch_size']]
            batch_features = process_batch(batch_paths)
            all_features.extend(batch_features)
            processed_files += len(batch_features)
            print(f"Processed {processed_files}/{CONFIG['max_files']} files.")
            if processed_files >= CONFIG['max_files']:
                break

    # Data standardization and PCA
    feature_vectors = np.array([np.concatenate((f['mfcc'], f['spectral_centroid'], f['zero_crossing_rate'])) for f in all_features if f is not None])
    scaler = StandardScaler()
    scaled_feature_vectors = scaler.fit_transform(feature_vectors)
    pca = PCA(n_components=min(50, scaled_feature_vectors.shape[1]))
    reduced_features = pca.fit_transform(scaled_feature_vectors)
    print("PCA transformation successful.")

    # K-Means clustering
    kmeans = KMeans(n_clusters=CONFIG['num_clusters'], random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_features)
    print("K-Means clustering completed.")

    # Insert into MongoDB
    for feature, pca_feature, label in zip(all_features, reduced_features, cluster_labels):
        feature['pca_features'] = pca_feature.tolist()
        feature['cluster_label'] = label
        converted_feature = convert_numpy_to_python(feature)
        collection.insert_one(converted_feature)
        print(f"Feature inserted into MongoDB for file: {feature['file_path']}.")

    client.close()

if __name__ == '__main__':
    main()

