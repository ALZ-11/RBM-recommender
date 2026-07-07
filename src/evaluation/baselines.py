# src/evaluation/baselines.py
import numpy as np
import torch
from src.evaluation.metrics import compute_precision_recall_ndcg, compute_rmse

class PopularityRecommender:
    def __init__(self, train_dataset):
        """
        Learns global item popularity strictly from the training dataset cohort.
        """
        # Exclude dense matrices by retrieving active item counts from get_cohort_summaries
        _, num_ratings, _ = train_dataset.get_cohort_summaries()
        self.popularity_scores = num_ratings

    def evaluate(self, dataloader, top_k=10):
        """
        Evaluates the popularity recommender on unseen test targets.
        """
        all_precisions = []
        all_recalls = []
        all_ndcgs = []
        
        scores_template = torch.from_numpy(self.popularity_scores).float()
        
        for v_input, m_input, v_target, m_target in dataloader:
            batch_size = v_input.size(0)
            
            # Broadcast popularity scores across the batch
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
        strictly from the training user cohort, using a Bayesian shrinkage prior.
        """
        # Exclude dense matrices by retrieving sum and count from get_cohort_summaries
        sum_ratings, num_ratings, global_mean = train_dataset.get_cohort_summaries()
        self.global_mean = global_mean
        
        # Bayesian Shrinkage Prior (k = pseudo-observations)
        k = 10.0
        self.movie_means = np.where(
            num_ratings > 0, 
            (sum_ratings + k * self.global_mean) / (num_ratings + k), 
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