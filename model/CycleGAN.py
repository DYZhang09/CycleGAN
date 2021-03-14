import torch
import torch.nn as nn
import itertools
from model.networks import *
from model.utils import ImagePool, WeightInitializer


###############################################################
# This file defines the high-level CycleGAN model including   #
# the losses and optimizers                                   #
###############################################################


class CycleGAN(nn.Module):
    def __init__(self, A_nc, B_nc, res_block_num=9,
                 init_type='normal', init_gain=0.02,
                 lr=0.0002, lambda_A=10, lambda_B=10, lambda_idt=0):
        """
        initialize the CycleGAN model

        :param A_nc: the image channels of domain A
        :param B_nc: the image channels of domain B
        :param res_block_num: the num of resnet blocks in generators
        :param lr: the learning rate
        :param lambda_A: the coefficient of cycle loss of generator(B to A)
        :param lambda_B: the coefficient of cycle loss of generator(A to B)
        :param lambda_idt: the coefficient of identity loss, if equals 0 then no identity loss
        """
        super(CycleGAN, self).__init__()
        self.G_a2b = Generator(in_nc=A_nc, out_nc=B_nc, res_num=res_block_num)
        self.G_b2a = Generator(in_nc=B_nc, out_nc=A_nc, res_num=res_block_num)
        self.D_a = Discriminator(in_nc=A_nc, out_nc=1)
        self.D_b = Discriminator(in_nc=B_nc, out_nc=1)
        self.weight_init = WeightInitializer(init_type=init_type, init_gain=init_gain)
        self.real_A = None
        self.real_B = None
        self.fake_A = None
        self.fake_B = None
        self.opti_G = torch.optim.Adam(itertools.chain(self.G_a2b.parameters(), self.G_b2a.parameters()), lr=lr)
        self.opti_D = torch.optim.Adam(itertools.chain(self.D_a.parameters(), self.D_b.parameters()), lr=lr)
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_idt = lambda_idt
        self.image_pool_A = ImagePool(pool_size=50)
        self.image_pool_B = ImagePool(pool_size=50)

        self.weight_init.init(self.G_a2b)
        self.weight_init.init(self.G_b2a)
        self.weight_init.init(self.D_a)
        self.weight_init.init(self.D_b)

    def set_input(self, real_A, real_B):
        """
        set the input of CycleGAN. the image size of domains must be the same.

        :param real_A: the real images of domain A
        :param real_B: the real images of domain B
        :return: None
        """
        assert real_A.shape[2:] == real_B.shape[2:]
        self.real_A = real_A
        self.real_B = real_B

    def set_require_grads(self, nets, is_req_grads=True):
        """
        set the net parameters require grads

        :param nets: nets need to be set up
        :param is_req_grads: whether the nets need grads
        :return: None
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = is_req_grads

    def get_fake_A(self):
        """
        get the fake images of domain A

        :return: fake images of domain A
        """
        return self.fake_A

    def get_fake_B(self):
        """
        get the fake images of domain B

        :return: fake images of domain B
        """
        return self.fake_B

    def gan_loss(self, prediction, is_tar_real):
        """
        calculate the gan loss given the prediction and the target

        :param prediction: the prediction given by discriminators
        :param is_tar_real: if wants the discriminator to predict images as real
        :return: the GAN loss
        """
        if is_tar_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        lossfunc = nn.MSELoss()
        loss = lossfunc(prediction, target)
        return loss

    def l1_loss(self, src, dst):
        """
        l1 loss

        :param src: input
        :param dst: input
        :return: l1 loss between the src and dst
        """
        lossfunc = nn.L1Loss()
        return lossfunc(src, dst)

    def cycle_loss(self, origin, transformed):
        """
        cycle loss without coefficient

        :param origin: the origin real images
        :param transformed: the images transformed by generators
        :return: the Cycle loss
        """
        return self.l1_loss(origin, transformed)

    def identity_loss(self, target, transformed):
        """
        identity loss without coefficient

        :param target: the target images
        :param transformed: the images transformed by generators
        :return: the Identity loss
        """
        return self.l1_loss(target, transformed)

    def forward(self):
        """
        standard forward. generates the fake images and reconstuction images in both domains.

        :return: None
        """
        self.fake_B = self.G_a2b(self.real_A)
        self.fake_A = self.G_b2a(self.real_B)
        self.rec_A = self.G_b2a(self.fake_B)
        self.rec_B = self.G_a2b(self.fake_A)

    def backward_D(self, net, real, fake):
        """
        calculate grads and loss of a discriminator

        :param net: the discriminator net
        :param real: the real imgaes
        :param fake: the fake images
        :return: the Gan loss of the discriminator
        """
        prediction = net(real)
        loss_real = self.gan_loss(prediction, True)
        prediction = net(fake.detach())
        loss_fake = self.gan_loss(prediction, False)
        loss = (loss_real + loss_fake) * 0.5
        loss.backward()
        return loss

    def backward_D_A(self):
        """
        calculate grads and loss of the discriminator of domain A

        :return: None
        """
        fake_A = self.image_pool_A.query(self.fake_A)
        self.loss_D_A = self.backward_D(self.D_a, self.real_A, fake_A)

    def backward_D_B(self):
        """
        calculate grads and loss of the discriminator of domain B

        :return: None
        """
        fake_B = self.image_pool_B.query(self.fake_B)
        self.loss_D_B = self.backward_D(self.D_b, self.real_B, fake_B)

    def backward_G(self):
        """
        calculate grads and loss of both generators

        :return: None
        """
        self.loss_idt_A = 0
        self.loss_idt_B = 0
        if self.lambda_idt > 0:
            self.loss_idt_A = self.identity_loss(self.real_B, self.fake_A) * self.lambda_B * self.lambda_idt
            self.loss_idt_B = self.identity_loss(self.real_A, self.fake_B) * self.lambda_A * self.lambda_idt

        self.loss_gan_a2b = self.gan_loss(self.D_b(self.fake_B), True)
        self.loss_gan_b2a = self.gan_loss(self.D_a(self.fake_A), True)
        self.loss_cycle_a = self.cycle_loss(self.real_A, self.rec_A) * self.lambda_A
        self.loss_cycle_b = self.cycle_loss(self.real_B, self.rec_B) * self.lambda_B
        self.loss_G = self.loss_idt_A + self.loss_idt_B + \
                      self.loss_gan_a2b + self.loss_gan_b2a + \
                      self.loss_cycle_a + self.loss_cycle_b
        self.loss_G.backward()

    def optimize(self):
        """
        optimize the parameters of discriminators and generators a step

        :return: None
        """
        self.forward()

        # optimize the generators
        self.set_require_grads([self.D_a, self.D_b], False)
        self.opti_G.zero_grad()
        self.backward_G()
        self.opti_G.step()

        # optimize the discriminators
        self.set_require_grads([self.D_a, self.D_b], True)
        self.opti_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.opti_D.step()

    def print_loss_info(self, epoch=None):
        info = ''
        if epoch is not None:
            info += '| epoch: %d |' % epoch
        info += ' l_gan_a2b %.6f | l_gan_b2a %.6f | l_cyc_a %.6f | l_cyc_b %.6f |' % (self.loss_gan_a2b,
                                                                                      self.loss_gan_b2a,
                                                                                      self.loss_cycle_a,
                                                                                      self.loss_cycle_b)
        info += ' l_D_a %.6f | l_D_b %.6f |' % (self.loss_D_A, self.loss_D_B)
        print(info)

    def test(self):
        """
        test the model without calculating the grads

        :return: None
        """
        with torch.no_grad():
            self.forward()


# briefly test the CycleGAN implementation
if __name__ == '__main__':
    N, C, H, W = 100, 3, 64, 64
    test_input_a = torch.rand((N, C, H, W))
    test_input_B = torch.rand((N, C, H, W))

    cycle_gan = CycleGAN(C, C)
    cycle_gan.set_input(test_input_a, test_input_B)
    cycle_gan.forward()
    print(cycle_gan.get_fake_A().shape, cycle_gan.get_fake_B().shape)

    cycle_gan.optimize()
    cycle_gan.test()
    print(cycle_gan.get_fake_A().shape, cycle_gan.get_fake_B().shape)
