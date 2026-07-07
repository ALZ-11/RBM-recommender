# src/models/gbrbm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianBernoulliRBM(nn.Module):
    def __init__(self, visible_units, hidden_units, sigma=0.1):
        """
        Initializes parameter tensors for the Masked Gaussian-Bernoulli RBM.
        Args:
            visible_units (int): Number of visible units D (movies)
            hidden_units (int): Number of hidden units F (latent features)
            sigma (float): Standard deviation of the Gaussian visible units
        """
        super(GaussianBernoulliRBM, self).__init__()
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.sigma = sigma
        
        # Initialize weights with standard guidelines (scaled random normal)
        self.W = nn.Parameter(torch.randn(hidden_units, visible_units) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_units))
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))

    def sample_h(self, v, mask):
        """
        Computes conditional probability P(h_j = 1 | v; m) and draws binary samples.
        Args:
            v (Tensor): Visible state tensor of shape (batch_size, visible_units)
            mask (Tensor): Mask state tensor of shape (batch_size, visible_units)
        Returns:
            p_h (Tensor): Activation probabilities of shape (batch_size, hidden_units)
            h_sample (Tensor): Bernoulli sampled states of shape (batch_size, hidden_units)
        """
        # Exclude unobserved visible units from contributing input to hidden activations
        v_masked = v * mask
        activation = F.linear(v_masked / self.sigma, self.W, self.h_bias)
        p_h = torch.sigmoid(activation)
        h_sample = torch.bernoulli(p_h)
        return p_h, h_sample

    def sample_v(self, h, mask):
        """
        Computes conditional expectation E[v_i | h; m] and draws reconstructions.
        Args:
            h (Tensor): Hidden state tensor of shape (batch_size, hidden_units)
            mask (Tensor): Mask state tensor of shape (batch_size, visible_units)
        Returns:
            v_mean_masked (Tensor): Continuous mean activations of shape (batch_size, visible_units)
            v_sample_masked (Tensor): Gaussian sampled states of shape (batch_size, visible_units)
        """
        # Calculate continuous conditional expectations
        v_mean = self.v_bias + self.sigma * F.linear(h, self.W.t())
        
        # Add normal Gaussian noise scaled by visible standard deviation
        noise = torch.randn_like(v_mean) * self.sigma
        v_sample = v_mean + noise
        
        # Apply mask constraint to zero out unobserved variables
        v_mean_masked = v_mean * mask
        v_sample_masked = v_sample * mask
        
        return v_mean_masked, v_sample_masked