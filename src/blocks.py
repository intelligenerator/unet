from typing import Union, Tuple

import torch
import torch.nn as nn

tuple_or_int = Union[int, Tuple[int, int]]


class ConvBlock(nn.Module):
    """U-Net Convolution block

    Parameters
    ----------
    in_channels, out_channels: int
        Number of input and desired output channels.
    kernel_size: `tuple` or int, optional
        Kernel size to use for the convolution. Default is `(3, 3)`.
    stride: `tuple` or int, optional
        Stride to use for the convolution. Default is `1`.
    depth: int, optional
        Amount of convolutions to perform. Default is `2`.
    activation: str, optional
        Activation to use after every convolution. If 'relu' is not specified,
        no activation will be used. Default is `'relu'`.
    batch_norm: bool, optional
        Whether or not to use batch normalization after the activation. Default
        is `True`.
    padding: int or bool, optional
        Padding to apply before convoluting. Default is `0`.

    Attributes
    ----------
    net: nn.Sequential
        Convolution block.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple_or_int = (3, 3), stride: tuple_or_int = 1,
                 depth: int = 2, activation: str = 'relu',
                 batch_norm: bool = True, padding: Union[int, bool] = 0):
        super(ConvBlock, self).__init__()

        blocks = []
        for i in range(depth):
            blocks.append(nn.Conv2d(in_channels if i == 0 else out_channels,
                                    out_channels, kernel_size, stride=stride,
                                    padding=int(padding)))
            if activation == 'relu':
                blocks.append(nn.ReLU())
            if batch_norm:
                blocks.append(nn.BatchNorm2d(out_channels))

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class DeconvBlock(nn.Module):
    """U-Net Deconvolution block

    Parameters
    ----------
    in_channels, out_channels: int
        Number of input and desired output channels.
    up: str, optional
        Mode of the upconvolution. Can be either 'upsample', applying upsampling
        to increase image dimensions, or 'upconv', applying a transposed
        convolution (`nn.ConvTranspose2d`). Default is `'upsample'`.
    scale_factor: `tuple` or int, optional
        Scale of the upscaled image compared to the input image. When applying
        transposed convolution, this will also be used as stride. Default is
        `(2, 2)`.
    mode: str, optional
        Upsampling algorithm. This only applies when using upsample mode. For
        4D tensors this may only be bilinear. Default is `'bilinear'`.
    align_corners: bool, optional
        Whether or not to align image corners when upsampling. This only applies
        when using upsampling mode. Default is `False`. 
    **kwargs
        Key-word arguments to be passed to the `ConvBlock` constructor when
        creating the convolution block.

    Attributes
    ----------
    up: nn.Sequential
        Upscaling block, using either upsampling or transposed convolution
    conv: ConvBlock
        Convolution block
    """

    def __init__(self, in_channels: int, out_channels: int,
                 up: str = 'upsample', scale_factor: tuple_or_int = (2, 2),
                 mode: str = 'bilinear', align_corners: bool = False,
                 **kwargs):
        super(DeconvBlock, self).__init__()
        assert up in ['upsample', 'upconv']

        if up == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode=mode, scale_factor=scale_factor,
                            align_corners=align_corners),
                nn.Conv2d(in_channels, out_channels, 1)
            )
        elif up == 'upconv':
            self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=scale_factor,
                                         stride=scale_factor)

        self.conv = ConvBlock(in_channels, out_channels, **kwargs)

    @staticmethod
    def center_crop(batch, target_size):
        """Center-crop image to specified size

        Parameters
        ----------
        batch: torch.Tensor
            4D input batch (NxCxHxW)
        target: tuple or list
            Target dimensions (HxW)
        """
        _, _, h, w = batch.size()
        target_h, target_w = target_size
        dh = (h - target_h) // 2
        dw = (w - target_w) // 2
        return batch[:, :, dh:(target_h+dh), dw:(target_w+dw)]

    def forward(self, accross, x):
        x = self.up(x)
        accross = self.center_crop(accross, x.shape[2:])
        x = torch.cat((accross, x), dim=1)

        return self.conv(x)
