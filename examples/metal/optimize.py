import mlx.core as mx  # noqa
import mlx.nn as nn
from typing import Union


class Model(nn.Module):
    """
    Model that performs a 2D convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Union[int, tuple]): Size of the convolution kernel.
        stride (Union[int, tuple]): Stride of the convolution. Default is 1.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def __call__(self, x):
        """
        Args:
            x (mx.array): Input tensor of shape (batch_size, height, width, in_channels).
        Returns:
            mx.array: Output tensor of shape (batch_size, height, width, out_channels).
        """
        return self.conv(x)
