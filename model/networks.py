import torch
import torch.nn as nn
from model.basic_blocks import *


#################################################################
# This file defines networks using the components from basic    #
# blocks.  Generators and Discriminators are defined here.      #
#################################################################

# ----------------------- Generator ---------------------------- #

class Generator(nn.Module):
    """
    the generator used in CycleGAN
    """

    def __init__(self, in_nc, out_nc=3, res_num=9):
        """
        initialize the generator used in CycleGAN

        :param in_nc: input channels
        :param out_nc: output channels
        :param res_num: the num of res blocks
        """
        super(Generator, self).__init__()
        dim = 64
        self.model = [
            nn.ReflectionPad2d(3),
            ConvNormReLUBlock(in_nc=in_nc, knum=dim, ksize=7, s=1),
            ConvNormReLUBlock(in_nc=dim, knum=2 * dim, ksize=3, s=2, pad=1),
            ConvNormReLUBlock(in_nc=2 * dim, knum=4 * dim, ksize=3, s=2, pad=1)
        ]
        for i in range(res_num):
            self.model += [ResBlock(knum=4 * dim)]
        self.model += [
            DeconvNormReLUBlock(in_nc=4 * dim, knum=2 * dim, ksize=3),
            DeconvNormReLUBlock(in_nc=2 * dim, knum=dim, ksize=3),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=dim, out_channels=out_nc, kernel_size=7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        """
        get output from generator

        :param input: the input of generator
        :return: the output of generator
        """
        return self.model(input)


class Discriminator(nn.Module):
    """
    the discriminator used in CycleGAN
    """

    def __init__(self, in_nc, out_nc=1):
        """
        initialize the discriminator used in CycleGAN

        :param in_nc: input channels
        :param out_nc: output channels
        """
        super(Discriminator, self).__init__()
        dim = 64
        self.model = [
            ConvNormReLUBlock(in_nc=in_nc, knum=dim, ksize=4, s=2, is_leaky=True, is_norm=False, pad=1)
        ]
        for i in range(1, 3):
            self.model += [
                ConvNormReLUBlock(in_nc=dim * 2 ** (i - 1), knum=dim * 2 ** i, ksize=4, s=2, is_leaky=True, pad=1)
            ]
        self.model += [
            ConvNormReLUBlock(in_nc=dim * 2 ** 2, knum=dim * 2 ** 3, ksize=4, s=1, is_leaky=True, pad=1),
            nn.Conv2d(in_channels=dim * 2 ** 3, out_channels=out_nc, kernel_size=4, padding=1)
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        """
        get output from discriminator
        :param input: input of discriminator
        :return: output of discriminator
        """
        return self.model(input)


# test the generator and discriminator implemented above
if __name__ == '__main__':
    N, C, H, W = 100, 3, 128, 128
    x = torch.rand((N, C, H, W))

    test_G = Generator(in_nc=C)
    res = test_G(x)
    print(res.shape)

    test_D = Discriminator(in_nc=C)
    res = test_D(x)
    print(res.shape)
