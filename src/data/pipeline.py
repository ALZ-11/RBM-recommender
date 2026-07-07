# src/data/pipeline.py
import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.dataset import MovieLensDataset

def load_and_split_data(config):
    """
    Loads raw MovieLens csv ratings, maps indices continuously, partitions users,
    and returns aligned Train and Test MovieLensDataset objects.
    """
    csv_path = config['dataset']['path']
    
    # Check if dataset directory exists, otherwise unpack the standard archive
    if not os.path.exists(csv_path):
        zip_path = "ml-latest-small.zip"
        if os.path.exists(zip_path):
            print(f"Unpacking {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('./')
        else:
            raise FileNotFoundError(
                f"Dataset file '{csv_path}' and archive '{zip_path}' were not found. "
                f"Please ensure 'ml-latest-small.zip' is in the root directory."
            )

    print("Loading ratings database...")
    ratings_df = pd.read_csv(csv_path)
    
    # Compile continuous unique mappings over the entire dataset footprint
    raw_user_to_idx = {raw_id: idx for idx, raw_id in enumerate(ratings_df['userId'].unique())}
    raw_movie_to_idx = {raw_id: idx for idx, raw_id in enumerate(ratings_df['movieId'].unique())}
    
    num_users = len(raw_user_to_idx)
    all_user_indices = np.arange(num_users)
    
    # Split users into training and testing sets
    train_indices, test_indices = train_test_split(
        all_user_indices,
        train_size=config['dataset']['train_split'],
        random_state=config['experiment']['seed']
    )
    
    print(f"Data pipeline complete: {len(train_indices)} train users, {len(test_indices)} test users.")
    
    # Instantiate training dataset
    train_dataset = MovieLensDataset(
        ratings_df=ratings_df,
        user_indices=train_indices,
        movie_mapping=raw_movie_to_idx,
        user_mapping=raw_user_to_idx,
        mode="train",
        test_ratio=config['dataset']['test_ratio'],
        seed=config['experiment']['seed']
    )
    
    # Instantiate testing dataset sharing identical index spaces
    test_dataset = MovieLensDataset(
        ratings_df=ratings_df,
        user_indices=test_indices,
        movie_mapping=raw_movie_to_idx,
        user_mapping=raw_user_to_idx,
        mode="test",
        test_ratio=config['dataset']['test_ratio'],
        seed=config['experiment']['seed']
    )
    
    return train_dataset, test_dataset