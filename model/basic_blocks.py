import torch
import torch.nn as nn


########################################################
# this file defines some basic layers used to build    #
# the generators and discriminators.                   #
########################################################


# ---------------------basic blocks definition--------------------------- #

class ConvNormReLUBlock(nn.Module):
    def __init__(self, in_nc, knum, ksize=7, s=1, pad=0, is_leaky=False, leaky_slope=0.2, is_norm=True):
        """
        initialize this custom block\n
        architecture as below:\n
        ksize * ksize * knum conv layer with stride s --- InstanceNorm layer --- ReLU / LeakyReLU

        :param in_nc: input channels
        :param knum: the num of conv kernels
        :param ksize: the size of conv kernels
        :param s: the stride of conv kernels
        :param pad: the padding size
        :param is_leaky: if True, replace ReLU with LeakyReLU
        :param leaky_slope: the slope of LeakyReLU
        :param is_norm: if False, not using InstanceNorm layer
        """
        super(ConvNormReLUBlock, self).__init__()
        self.model = [nn.Conv2d(in_channels=in_nc, out_channels=knum,
                                kernel_size=ksize, stride=s, padding=pad)]
        if is_norm:
            self.model += [nn.InstanceNorm2d(num_features=knum)]
        if not is_leaky:
            self.model += [nn.ReLU(True)]
        else:
            self.model += [nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)]
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        """
        standard forward

        :param input: the input of block
        :return: the output of block
        """
        return self.model(input)


class ResBlock(nn.Module):
    def __init__(self, knum):
        """
        initialize the block
        the Resnet block:\n
        (ksize * ksize * knum conv layer with stride s) * 2 --- skip connection

        :param knum: the num of conv kernels
        """
        super(ResBlock, self).__init__()
        self.model = [nn.Conv2d(in_channels=knum, out_channels=knum,
                                kernel_size=3, stride=1, padding=1)] * 2
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        """
        standard forward

        :param input: input of block
        :return: output of block
        """
        return input + self.model(input)


class DeconvNormReLUBlock(nn.Module):
    def __init__(self, in_nc, knum, ksize, s=2, pad=1, output_pad=1):
        """
        initialize the block\n
        architecture as below:\n
        ksize * ksize * knum deconv2d layer with stride 1/2 --- InstanceNorm --- ReLU

        :param in_nc: input channels
        :param knum: the num of deconv kernels
        :param ksize: the size of deconv kernels
        :param s: stride
        :param pad: input padding size
        :param output_pad: output padding size
        """
        super(DeconvNormReLUBlock, self).__init__()
        self.model = [nn.ConvTranspose2d(in_channels=in_nc, out_channels=knum,
                                         kernel_size=ksize, stride=s,
                                         padding=pad, output_padding=output_pad)]
        self.model += [nn.InstanceNorm2d(num_features=knum)]
        self.model += [nn.ReLU(True)]
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        """
        standard forward

        :param input: input of block
        :return: output of block
        """
        return self.model(input)


# ------------------------------------------------------------------------------- #

# test the blocks defined above
if __name__ == '__main__':
    N = 100
    C = 3
    H = 128
    W = 128
    x = torch.rand((N, C, H, W))

    ksize = 7
    knum = 64
    stride = 2
    test_conv_block = ConvNormReLUBlock(in_nc=3, knum=64, ksize=7, s=stride)
    res = test_conv_block(x)
    print(res.shape)

    test_res_block = ResBlock(knum=knum)
    res = test_res_block(res)
    print(res.shape)

    test_deconv_block = DeconvNormReLUBlock(in_nc=knum, knum=C, ksize=ksize)
    res = test_deconv_block(res)
    print(res.shape)
