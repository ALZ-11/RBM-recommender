# src/evaluation/evaluator.py
import torch
import numpy as np
from src.evaluation.metrics import compute_rmse, compute_precision_recall_ndcg

@torch.no_grad()
def evaluate_model(model, dataloader, top_k=10, device="cpu"):
    """
    Evaluates the GBRBM on unseen target metrics without introducing data leakage.
    """
    model.eval()
    all_rmse = []
    all_precisions = []
    all_recalls = []
    all_ndcgs = []
    
    for v_input, m_input, v_target, m_target in dataloader:
        v_input = v_input.to(device)
        m_input = m_input.to(device)
        v_target = v_target.to(device)
        m_target = m_target.to(device)
        
        # Forward pass using Context (Input History) only
        _, h_sample = model.sample_h(v_input, m_input)
        
        # Reconstruct visible units (continuous expected mean)
        # pass a mask of all ones so the model reconstructs the entire space
        full_mask = torch.ones_like(m_input)
        v_recon_mean, _ = model.sample_v(h_sample, full_mask)
        
        # Calculate prediction RMSE strictly on targets
        rmse = compute_rmse(v_recon_mean, v_target, m_target)
        if rmse is not None:
            all_rmse.append(rmse)
            
        # Calculate ranking performance strictly on targets
        precision, recall, ndcg = compute_precision_recall_ndcg(
            v_recon_mean, v_input, m_input, v_target, m_target, top_k
        )
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_ndcgs.append(ndcg)
        
    model.train()
    
    return {
        "rmse": np.mean(all_rmse) if len(all_rmse) > 0 else np.nan,
        "precision_at_k": np.mean(all_precisions),
        "recall_at_k": np.mean(all_recalls),
        "ndcg_at_k": np.mean(all_ndcgs)
    }