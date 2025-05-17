import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Model that performs a matrix multiplication, summation, and combined scaling.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        # Combine the division by 2 and the scaling factor into one operation
        self.effective_scaling_factor = scaling_factor / 2.0

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        # Original operations: matmul -> / 2 -> sum -> * scaling_factor
        # Optimized operations: matmul -> sum -> * (scaling_factor / 2)
        x = torch.matmul(x, self.weight.T)
        x = torch.sum(x, dim=1, keepdim=True)
        x = x * self.effective_scaling_factor
        return x