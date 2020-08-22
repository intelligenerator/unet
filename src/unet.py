import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, DeconvBlock


class UNet(nn.Module):
    """Simple PyTorch U-Net

    Leaving all default parameters will create a U-Net as described in the
    original paper:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (Ronneberger, et al.)

    Parameters
    ----------
    in_channels: int, optional
        Number C if input channels in a batch of ``(N x C x H x W)``. Defaults
        to ``1`` corresponding to a  standard black-and-white image.
    out_channels: int, optional
        Number C of output channels in a batch of ``(N x C x H x W)``.
        Corresponds to the amount of classes that should be differentiated, e.g.
        C = 10 if you want to differentiate between 10 different segmentation
        classes. Defaults to ``2``.
    depth: int, optional
        Depth of the U-Net. A depth of ``5`` corresponds to the U-Net described
        in the original paper, i.e. ``5`` convolutional layers and ``4``
        up-convolutional layers. Default is ``5``.
    start_features: int, optional
        Number of initial features in the convolutional blocks. Features are
        doubled (or halved when up-convoluting) after every block. Default is
        ``64``.
    up: str, optional
        Mode of the up-convolution. May be either ``'upsample'`` or
        ``'upconv'``. Default is ``'upsample'``.
    padding: int, optional
        Padding to be applied when convoluting. User must ensure validity.
        Default is ``0``.
    verbose: bool, optional
        Verbose mode. Whether or not to print debug information about the batch
        size after every convolution step. Useful for fixing image size and
        padding problems. Default is ``False``.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2,
                 depth: int = 5, start_features: int = 64,
                 up: str = 'upsample', padding: int = 0,
                 verbose: bool = False):
        super(UNet, self).__init__()

        self.verbose = verbose
        curr_features = start_features

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        self.down.append(ConvBlock(in_channels, curr_features,
                                   padding=int(padding)))
        for _ in range(depth-1):
            self.down.append(
                ConvBlock(curr_features, curr_features*2, padding=padding))
            curr_features *= 2

        for _ in reversed(range(depth-1)):
            self.up.append(
                DeconvBlock(curr_features, curr_features //
                            2, up=up, padding=padding)
            )
            curr_features //= 2

        self.out = nn.Conv2d(curr_features, out_channels, 1)

    def forward(self, x):
        mem = []
        if self.verbose:
            in_shape = x.shape

        for i, conv in enumerate(self.down):
            x = conv(x)
            if i < (len(self.down) - 1):
                mem.append(x)
                x = F.max_pool2d(x, (2, 2))

            if self.verbose:
                print("[↓]", in_shape, "->", x.shape, end="\n\n")
                in_shape = x.shape

        for i, deconv in enumerate(self.up):
            accross = mem.pop()
            x = deconv(accross, x)
            if self.verbose:
                print("[↑]", in_shape, "->", x.shape,
                      "  accross:", accross.shape, end="\n\n")
                in_shape = x.shape

        return self.out(x)
