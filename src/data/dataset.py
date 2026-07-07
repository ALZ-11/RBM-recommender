# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class MovieLensDataset(Dataset):
    def __init__(self, ratings_df, user_indices, movie_mapping=None, user_mapping=None, mode="train", test_ratio=0.2, seed=42):
        """
        Args:
            ratings_df (pd.DataFrame): Cleaned ratings DataFrame.
            user_indices (list/np.ndarray): Global user indices assigned to this split.
            movie_mapping (dict, optional): Mapping dict from raw movieId to movie_idx.
            user_mapping (dict, optional): Mapping dict from raw userId to user_idx.
            mode (str): "train" or "test".
            test_ratio (float): Ratio of ratings to hold out for test user evaluation (gamma).
            seed (int): Base random seed for partitioning.
        """
        self.mode = mode
        self.test_ratio = test_ratio
        self.seed = seed
        self.user_indices = user_indices

        # Re-use existing indices maps if provided to keep train/test spaces aligned
        if user_mapping is None:
            self.raw_user_to_idx = {raw_id: idx for idx, raw_id in enumerate(ratings_df['userId'].unique())}
        else:
            self.raw_user_to_idx = user_mapping
            
        if movie_mapping is None:
            self.raw_movie_to_idx = {raw_id: idx for idx, raw_id in enumerate(ratings_df['movieId'].unique())}
        else:
            self.raw_movie_to_idx = movie_mapping
            
        self.num_users = len(self.raw_user_to_idx)
        self.num_movies = len(self.raw_movie_to_idx)

        # Memory-efficient sparse storage (fix)
        # store ratings in a list of dictionaries instead of allocating massive dense 2D matrices
        self.user_ratings = [{} for _ in range(self.num_users)]

        for row in ratings_df.itertuples():
            u_idx = self.raw_user_to_idx[row.userId]
            m_idx = self.raw_movie_to_idx[row.movieId]
            # Global linear scale: maps raw [0.5, 5.0] to [0.0, 1.0]
            self.user_ratings[u_idx][m_idx] = (row.rating - 0.5) / 4.5

    def __len__(self):
        return len(self.user_indices)

    def get_cohort_summaries(self):
        """
        Computes the sum of ratings, sum of masks, and global mean strictly over 
        the active user_indices cohort without constructing a global dense matrix.
        Returns:
            sum_ratings (ndarray): Shape (num_movies,)
            num_ratings (ndarray): Shape (num_movies,)
            global_mean (float): Scalar global average
        """
        sum_ratings = np.zeros(self.num_movies, dtype=np.float32)
        num_ratings = np.zeros(self.num_movies, dtype=np.float32)
        
        for user_idx in self.user_indices:
            for m_idx, rating in self.user_ratings[user_idx].items():
                sum_ratings[m_idx] += rating
                num_ratings[m_idx] += 1.0
                
        total_sum = np.sum(sum_ratings)
        total_count = np.sum(num_ratings)
        global_mean = total_sum / max(1.0, total_count)
        
        return sum_ratings, num_ratings, global_mean

    def _get_user_dense_vectors(self, user_idx):
        """Helper to reconstruct dense representations on-the-fly for a user."""
        v_user = np.zeros(self.num_movies, dtype=np.float32)
        m_user = np.zeros(self.num_movies, dtype=np.float32)
        
        for m_idx, rating in self.user_ratings[user_idx].items():
            v_user[m_idx] = rating
            m_user[m_idx] = 1.0
        return v_user, m_user

    def __getitem__(self, idx):
        # Translate dataset-relative index back to the global index space
        user_idx = self.user_indices[idx]
        v_user, m_user = self._get_user_dense_vectors(user_idx)

        if self.mode == "train":
            return torch.from_numpy(v_user), torch.from_numpy(m_user)

        elif self.mode == "test":
            # Retrieve movie indices rated by this user directly from dictionary keys
            rated_indices = np.array(list(self.user_ratings[user_idx].keys()))
            num_ratings = len(rated_indices)

            # Initialize user-specific seed for deterministic splitting (Phase 2, Step 3)
            rng = np.random.default_rng(self.seed + user_idx)
            shuffled_indices = rng.permutation(rated_indices)

            k = max(1, int((1.0 - self.test_ratio) * num_ratings))
            input_indices = shuffled_indices[0:k]
            target_indices = shuffled_indices[k:]

            # Generate observable input representation (history)
            m_input = np.zeros_like(m_user)
            m_input[input_indices] = 1.0
            v_input = v_user * m_input

            # Generate held-out target representation (evaluation ground truth)
            m_target = np.zeros_like(m_user)
            m_target[target_indices] = 1.0
            v_target = v_user * m_target

            return (
                torch.from_numpy(v_input),
                torch.from_numpy(m_input),
                torch.from_numpy(v_target),
                torch.from_numpy(m_target)
            )