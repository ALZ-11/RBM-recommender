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
    and returns aligned Train and Test MovieLensDataset objects, excluding item-level cold starts.
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
    
    # Split raw user IDs before indexing (to ensure clean separation of cohorts)
    raw_users = ratings_df['userId'].unique()
    train_users, test_users = train_test_split(
        raw_users,
        train_size=config['dataset']['train_split'],
        random_state=config['experiment']['seed']
    )
    
    # Identify movies that have at least one rating in the training user set
    train_ratings_subset = ratings_df[ratings_df['userId'].isin(train_users)]
    trained_movie_ids = train_ratings_subset['movieId'].unique()
    
    # Filter the global ratings dataframe to exclude cold items (never seen in training)
    ratings_df_filtered = ratings_df[ratings_df['movieId'].isin(trained_movie_ids)].copy()
    
    # Compile continuous mappings over the filtered dataset footprint
    raw_user_to_idx = {raw_id: idx for idx, raw_id in enumerate(ratings_df_filtered['userId'].unique())}
    raw_movie_to_idx = {raw_id: idx for idx, raw_id in enumerate(ratings_df_filtered['movieId'].unique())}
    
    # Map the split user IDs back to their continuous index space
    train_indices = np.array([raw_user_to_idx[uid] for uid in train_users if uid in raw_user_to_idx])
    test_indices = np.array([raw_user_to_idx[uid] for uid in test_users if uid in raw_user_to_idx])
    
    print(f"Data pipeline complete: {len(train_indices)} train users, {len(test_indices)} test users.")
    print(f"Total active movies: {len(raw_movie_to_idx)} (filtered {len(ratings_df['movieId'].unique()) - len(raw_movie_to_idx)} cold items).")
    
    # Instantiate training dataset using filtered data
    train_dataset = MovieLensDataset(
        ratings_df=ratings_df_filtered,
        user_indices=train_indices,
        movie_mapping=raw_movie_to_idx,
        user_mapping=raw_user_to_idx,
        mode="train",
        test_ratio=config['dataset']['test_ratio'],
        seed=config['experiment']['seed']
    )
    
    # Instantiate testing dataset sharing identical index spaces using filtered data
    test_dataset = MovieLensDataset(
        ratings_df=ratings_df_filtered,
        user_indices=test_indices,
        movie_mapping=raw_movie_to_idx,
        user_mapping=raw_user_to_idx,
        mode="test",
        test_ratio=config['dataset']['test_ratio'],
        seed=config['experiment']['seed']
    )
    
    return train_dataset, test_dataset