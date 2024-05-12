import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import os
import warnings
import logging
import random

# Suppress all warnings (Python warnings and DeprecationWarnings)
warnings.filterwarnings("ignore")

# Configure Python logging
logging.getLogger().setLevel(logging.ERROR)  # Suppress all logs except ERROR

# Set environment variables to control log4j logging
os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.ui.showConsoleProgress=false --driver-java-options=-Dlog4j.logLevel=error pyspark-shell'

# Import PySpark only after setting the log environment variables
from pyspark.sql import SparkSession

# Initialize Spark with minimal logging
spark = SparkSession.builder \
    .appName("Modeling") \
    .config("spark.pyspark.python", "/usr/bin/python3") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Define a PyTorch Dataset for our song features
class SongFeaturesDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32)

# Neural Network Model for learning song embeddings
class FeatureEmbeddingModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        print("Fetching...")
        super(FeatureEmbeddingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, input_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def connect_to_mongodb():
    client = MongoClient('localhost', 27017)
    db = client['feature_extract']
    collection = db['extraction']
    print("Fetching...")
    return collection

def get_song_features():
    collection = connect_to_mongodb()
    features = list(collection.find({}, {'_id': 0, 'mfcc': 1, 'file_path': 1}))
    return features

def train_embedding_model(features):
    print("Fetching...")
    dataset = SongFeaturesDataset([f['mfcc'] for f in features])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = FeatureEmbeddingModel(input_dim=len(features[0]['mfcc']))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        for data in dataloader:
            optimizer.zero_grad()
            data = data.float()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
    return model

def recommend_songs(model, features, current_song_index, top_n=5):
    print("Fetching...")
    embeddings = model(torch.tensor([f['mfcc'] for f in features], dtype=torch.float32)).detach().numpy()
    current_song_embedding = embeddings[current_song_index].reshape(1, -1)
    similarities = cosine_similarity(current_song_embedding, embeddings).flatten()
    similarities[current_song_index] = -np.inf

    top_indices = np.argsort(-similarities)[:top_n + 10]
    recommended_songs = []
    for index in top_indices:
        if features[index]['file_path'] not in recommended_songs and len(recommended_songs) < top_n:
            recommended_songs.append(features[index]['file_path'])

    return recommended_songs

def main():
    song_features = get_song_features()
    model = train_embedding_model(song_features)
    current_song_index = random.randint(0, len(song_features)-1)  
    print("Fetching...")
    recommended_songs = recommend_songs(model, song_features, current_song_index, top_n=5)
    print("Recommended Songs:")
    for song in recommended_songs:
        print(song)
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()

