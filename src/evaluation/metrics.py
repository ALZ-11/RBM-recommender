# src/evaluation/metrics.py
import numpy as np
import torch

def compute_rmse(v_recon, v_target, m_target):
    """
    Computes Root Mean Squared Error (RMSE) over the observed target ratings,
    scaled back to the original [0.5, 5.0] star rating scale.
    """
    recon = v_recon.detach().cpu().numpy()
    target = v_target.detach().cpu().numpy()
    mask = m_target.detach().cpu().numpy()
    
    num_targets = np.sum(mask)
    if num_targets == 0:
        return None
        
    sq_error = ((recon - target) * mask) ** 2
    rmse_normalized = np.sqrt(np.sum(sq_error) / num_targets)
    
    # Scale normalized error [0.0, 1.0] back to 5-star interval [0.5, 5.0]
    # Scale factor: (5.0 - 0.5) = 4.5
    rmse_original_scale = rmse_normalized * 4.5
    return rmse_original_scale

def compute_precision_recall_ndcg(v_recon, v_input, m_input, v_target, m_target, top_k=10, relevance_threshold=0.667):
    """
    Computes Precision@K, Recall@K, and NDCG@K strictly on unseen targets.
    Filters out any items present in the user's input history.
    
    relevance_threshold: Mapped normalized value corresponding to raw rating >= 3.5 stars
                        (e.g., (3.5 - 0.5) / 4.5 = 0.667)
    """
    recon = v_recon.detach().cpu().numpy()
    input_mask = m_input.detach().cpu().numpy()
    target_val = v_target.detach().cpu().numpy()
    target_mask = m_target.detach().cpu().numpy()
    
    batch_size = recon.shape[0]
    precisions = []
    recalls = []
    ndcgs = []
    
    for u in range(batch_size):
        # Mask out items that were in input history (set to highly negative score)
        scores = recon[u].copy()
        scores[input_mask[u] == 1.0] = -1e9
        
        # Get the indices of the top-K recommended items
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        # Identify relevant items in the target set
        relevant_indices = np.where((target_mask[u] == 1.0) & (target_val[u] >= relevance_threshold))[0]
        num_relevant = len(relevant_indices)
        
        # Decouple metric tracking to eliminate evaluation selection bias (fix)
        # If the user has no relevant target items in their holdout, hits are mathematically 0.
        if num_relevant == 0:
            precisions.append(0.0)
            ndcgs.append(0.0)
            # Recall is omitted to prevent division-by-zero
            continue
            
        # Calculate Precision & Recall
        hits = np.isin(top_k_indices, relevant_indices)
        num_hits = np.sum(hits)
        
        precision = num_hits / top_k
        recall = num_hits / num_relevant
        
        precisions.append(precision)
        recalls.append(recall)
        
        # Calculate NDCG
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            if idx in relevant_indices:
                dcg += 1.0 / np.log2(i + 2)
                
        idcg = 0.0
        for i in range(min(top_k, num_relevant)):
            idcg += 1.0 / np.log2(i + 2)
            
        ndcg = dcg / idcg if idcg > 0.0 else 0.0
        ndcgs.append(ndcg)
        
    # Compute cohort averages strictly over their respective valid distributions
    mean_precision = np.mean(precisions) if len(precisions) > 0 else 0.0
    mean_recall = np.mean(recalls) if len(recalls) > 0 else 0.0
    mean_ndcg = np.mean(ndcgs) if len(ndcgs) > 0 else 0.0
    
    return mean_precision, mean_recall, mean_ndcg