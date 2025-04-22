import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    This version uses F.linear for the matrix multiplication, followed by
    summation and scaling, avoiding torch.einsum.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        # Weight for F.linear: (out_features, in_features) -> (hidden_size, input_size)
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        # Store scaling_factor
        self.scaling_factor = scaling_factor
        # Precompute the combined scaling factor for efficiency
        self.combined_scale = self.scaling_factor / 2.0

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Original operation: sum( (x @ W.T) / 2 , dim=1) * scale
        # Equivalent operation: sum( F.linear(x, W) , dim=1) * (scale / 2)

        # 1. Compute x @ W.T using F.linear
        # x shape: (batch_size, input_size)
        # weight shape: (hidden_size, input_size)
        # linear_out shape: (batch_size, hidden_size)
        linear_out = F.linear(x, self.weight)

        # 2. Sum along the hidden_size dimension (dim=1)
        # summed_out shape: (batch_size,)
        summed_out = torch.sum(linear_out, dim=1)

        # 3. Apply the combined scaling factor
        # scaled_out shape: (batch_size,)
        scaled_out = summed_out * self.combined_scale

        # 4. Unsqueeze the last dimension to get shape (batch_size, 1)
        output = scaled_out.unsqueeze(1)

        return output
