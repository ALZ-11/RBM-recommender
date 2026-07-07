# src/evaluation/baselines.py
import numpy as np
import torch
from src.evaluation.metrics import compute_precision_recall_ndcg, compute_rmse

class PopularityRecommender:
    def __init__(self, train_dataset):
        """
        Learns global item popularity strictly from the training dataset cohort.
        """
        # Slice global_m using train_dataset.user_indices to exclude test user interactions
        train_masks = train_dataset.global_m[train_dataset.user_indices]
        self.popularity_scores = np.sum(train_masks, axis=0)

    def evaluate(self, dataloader, top_k=10):
        """
        Evaluates the popularity recommender on unseen test targets.
        """
        all_precisions = []
        all_recalls = []
        all_ndcgs = []
        
        # Expand popularity scores to match batch processing
        scores_template = torch.from_numpy(self.popularity_scores).float()
        
        for v_input, m_input, v_target, m_target in dataloader:
            batch_size = v_input.size(0)
            
            # Broadcast popularity scores across the batch
            # Shape: (batch_size, visible_units)
            v_recon_mean = scores_template.unsqueeze(0).expand(batch_size, -1)
            
            # Calculate ranking metrics
            precision, recall, ndcg = compute_precision_recall_ndcg(
                v_recon_mean, v_input, m_input, v_target, m_target, top_k
            )
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_ndcgs.append(ndcg)
            
        return {
            "rmse": np.nan,  # Popularity cannot predict explicit rating values
            "precision_at_k": np.mean(all_precisions) if len(all_precisions) > 0 else 0.0,
            "recall_at_k": np.mean(all_recalls) if len(all_recalls) > 0 else 0.0,
            "ndcg_at_k": np.mean(all_ndcgs) if len(all_ndcgs) > 0 else 0.0
        }


class GlobalMeanRecommender:
    def __init__(self, train_dataset):
        """
        Computes the global average rating and movie-specific average ratings
        strictly from the training user cohort.
        """
        # Slice global tensors using train_dataset.user_indices to isolate the training cohort
        train_ratings = train_dataset.global_v[train_dataset.user_indices]
        train_masks = train_dataset.global_m[train_dataset.user_indices]
        
        # Calculate individual movie averages over training users
        sum_ratings = np.sum(train_ratings, axis=0)
        num_ratings = np.sum(train_masks, axis=0)
        
        # Compute global average strictly over the training cohort
        global_sum = np.sum(train_ratings)
        global_count = np.sum(train_masks)
        self.global_mean = global_sum / max(1.0, global_count)
        
        # Avoid division by zero for unrated movies in train set by defaulting to global mean
        self.movie_means = np.where(
            num_ratings > 0, 
            sum_ratings / np.clip(num_ratings, 1.0, None), 
            self.global_mean
        )

    def evaluate(self, dataloader, top_k=10):
        """
        Evaluates the continuous rating prediction capability using training cohort averages.
        """
        all_rmse = []
        all_precisions = []
        all_recalls = []
        all_ndcgs = []
        
        means_template = torch.from_numpy(self.movie_means).float()
        
        for v_input, m_input, v_target, m_target in dataloader:
            batch_size = v_input.size(0)
            v_recon_mean = means_template.unsqueeze(0).expand(batch_size, -1)
            
            # Calculate prediction error (RMSE)
            rmse = compute_rmse(v_recon_mean, v_target, m_target)
            if rmse is not None:
                all_rmse.append(rmse)
                
            # Calculate ranking metrics
            precision, recall, ndcg = compute_precision_recall_ndcg(
                v_recon_mean, v_input, m_input, v_target, m_target, top_k
            )
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_ndcgs.append(ndcg)
            
        return {
            "rmse": np.mean(all_rmse) if len(all_rmse) > 0 else np.nan,
            "precision_at_k": np.mean(all_precisions) if len(all_precisions) > 0 else 0.0,
            "recall_at_k": np.mean(all_recalls) if len(all_recalls) > 0 else 0.0,
            "ndcg_at_k": np.mean(all_ndcgs) if len(all_ndcgs) > 0 else 0.0
        }