# src/training/trainer.py
import torch

class GBRBMTrainer:
    def __init__(self, model, optimizer, l2_reg=0.0002, cd_steps=1):
        """
        Args:
            model (GaussianBernoulliRBM): Instantiated GBRBM module
            optimizer (torch.optim.Optimizer): PyTorch optimizer
            l2_reg (float): L2 regularization (weight decay) coefficient
            cd_steps (int): Number of Gibbs sampling iterations (k)
        """
        self.model = model
        self.optimizer = optimizer
        self.l2_reg = l2_reg
        self.cd_steps = cd_steps

    def train_step(self, batch_v, batch_m):
        """
        Executes a single batch update using Masked Contrastive Divergence (CD-k).
        Args:
            batch_v (Tensor): Normalised rating tensor of shape (batch_size, visible_units)
            batch_m (Tensor): Binary observation mask of shape (batch_size, visible_units)
        Returns:
            reconstruction_error (float): Mean Squared Error calculated over observed indices only
        """
        batch_size = batch_v.size(0)
        self.optimizer.zero_grad()
        
        # Positive Phase
        pos_h_prob, pos_h_sample = self.model.sample_h(batch_v, batch_m)
        
        # Negative Phase (Gibbs Chain)
        v_curr = batch_v
        h_curr_sample = pos_h_sample
        v_curr_mean = None
        neg_h_prob = None
        
        for step in range(self.cd_steps):
            # Track both continuous expectations and sampled states
            v_curr_mean, v_curr = self.model.sample_v(h_curr_sample, batch_m)
            neg_h_prob, h_curr_sample = self.model.sample_h(v_curr, batch_m)
            
        # Calculate batch-level active mask for items (fix)
        # Shape: (visible_units,)
        item_active = (torch.sum(batch_m, dim=0) > 0).float()
            
        # Compute Manual Masked Gradients
        # Weights Gradient (with unobserved columns zeroed out and L2 regularization applied)
        pos_grad_W = torch.mm(pos_h_prob.t(), batch_v / self.model.sigma)
        neg_grad_W = torch.mm(neg_h_prob.t(), v_curr_mean / self.model.sigma)
        
        # Apply weight decay strictly to active items in this batch (fix)
        grad_W = -(pos_grad_W - neg_grad_W) / batch_size + self.l2_reg * self.model.W * item_active.unsqueeze(0)
        
        # Visible Bias Gradient (Masked and scaled by sigma^2)
        grad_v_bias = -torch.mean(((batch_v - v_curr_mean) * batch_m) / (self.model.sigma ** 2), dim=0)
        
        # Hidden Bias Gradient
        grad_h_bias = -torch.mean(pos_h_prob - neg_h_prob, dim=0)
        
        # Assign calculated gradients directly to parameter fields
        self.model.W.grad = grad_W
        self.model.v_bias.grad = grad_v_bias
        self.model.h_bias.grad = grad_h_bias
        
        # Stabilize updates by clipping gradient norms
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        
        # Calculate Observed Reconstruction Error (MSE over active ratings)
        observed_sq_error = ((batch_v - v_curr_mean) * batch_m) ** 2
        reconstruction_error = torch.sum(observed_sq_error) / torch.clamp(torch.sum(batch_m), min=1.0)
        
        return reconstruction_error.item()