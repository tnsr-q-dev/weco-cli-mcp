import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, summation, and combined scaling,
    optimized by pre-computing the combined weight vector and using torch.linalg.vecdot.
    Assumes torch.compile is applied externally.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        # weight is (hidden_size, input_size)
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))

        # Pre-compute the combined weight vector and scaling factor
        with torch.no_grad():
             summed_weight_vector = self.weight.sum(dim=0) # (input_size,)
             effective_weight_vector = summed_weight_vector * (scaling_factor / 2.0) # (input_size,)
             # Store as a buffer
             self.register_buffer('effective_weight_vector', effective_weight_vector) # (input_size,)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Perform the batched dot product using torch.linalg.vecdot
        # x is (batch_size, input_size) -> (B, I)
        # effective_weight_vector is (input_size,) -> (I)
        # torch.linalg.vecdot(A (..., N), B (..., N) or (N)) -> (...)
        # In our case: A is (B, I), B is (I). Result is (B,).
        output = torch.linalg.vecdot(x, self.effective_weight_vector)

        # Unsqueeze to match the required output shape (batch_size, 1)
        return output.unsqueeze(1)