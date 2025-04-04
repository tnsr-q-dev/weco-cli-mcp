import mlx.core as mx
import mlx.nn as nn


class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = mx.random.normal(shape=(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def __call__(self, x):
        """
        Args:
            x (mx.array): Input tensor of shape (batch_size, input_size).
        Returns:
            mx.array: Output tensor of shape (batch_size, hidden_size).
        """
        x = mx.matmul(x, mx.transpose(self.weight))
        x = x / 2
        x = mx.sum(x, axis=1, keepdims=True)
        x = x * self.scaling_factor
        return x
