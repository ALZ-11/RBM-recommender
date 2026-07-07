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
            batch_v (Tensor): Normalized rating tensor of shape (batch_size, visible_units)
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
            
            # Clip expectations and sampled states to [0.0, 1.0] to respect target bounds
            v_curr_mean = torch.clamp(v_curr_mean, 0.0, 1.0)
            v_curr = torch.clamp(v_curr, 0.0, 1.0)
            
            neg_h_prob, h_curr_sample = self.model.sample_h(v_curr, batch_m)
            
        # Run a final expectation step to align the visible reconstruction at step k
        v_final_mean, _ = self.model.sample_v(h_curr_sample, batch_m)
        v_final_mean = torch.clamp(v_final_mean, 0.0, 1.0)
            
        # Calculate batch-level active mask for items
        item_active = (torch.sum(batch_m, dim=0) > 0).float()
            
        # Compute Manual Masked Gradients
        pos_grad_W = torch.mm(pos_h_prob.t(), batch_v)
        neg_grad_W = torch.mm(neg_h_prob.t(), v_final_mean)
        
        # Restore the formal 1/sigma scaling to the weight gradients (fix)
        grad_W = (-(pos_grad_W - neg_grad_W) / batch_size) / self.model.sigma
        
        # Restore the formal 1/sigma^2 scaling to the visible bias gradients (fix)
        grad_v_bias = -torch.mean((batch_v - v_final_mean) * batch_m, dim=0) / (self.model.sigma ** 2)
        
        # Hidden Bias Gradient
        grad_h_bias = -torch.mean(pos_h_prob - neg_h_prob, dim=0)
        
        # Assign calculated gradients directly to parameter fields
        self.model.W.grad = grad_W
        self.model.v_bias.grad = grad_v_bias
        self.model.h_bias.grad = grad_h_bias
        
        # Stabilize updates by clipping gradient norms
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        
        self.optimizer.step()
        
        # True Decoupled Weight Decay (AdamW-Style) (fix)
        # Apply L2 decay directly to W post-update, restricted exclusively to active items.
        # (avoids Adam's moment-tracking scaling distortions)
        if self.l2_reg > 0:
            with torch.no_grad():
                current_lr = self.optimizer.param_groups[0]['lr']
                # W = W * (1 - lr * l2_reg * active_mask)
                self.model.W.mul_(1.0 - current_lr * self.l2_reg * item_active.unsqueeze(0))
        
        # Calculate observed reconstruction error (MSE over active ratings)
        observed_sq_error = ((batch_v - v_final_mean) * batch_m) ** 2
        reconstruction_error = torch.sum(observed_sq_error) / torch.clamp(torch.sum(batch_m), min=1.0)
        
        return reconstruction_error.item()