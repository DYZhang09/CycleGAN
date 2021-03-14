import torch
import numpy
import os
import argparse
import PIL.Image as Image
import torch.nn as nn

from dataIO.datasets import UnalignedDataset, Dataloader, Imgwriter
from dataIO.preprocess import Transform, TransParams
from dataIO.utils import fake_img_paths, mkdir
from model.CycleGAN import CycleGAN

##################################################
# This file defines the training code.           #
##################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test the CycleGAN after training.')
    parser.add_argument('--dataroot', required=True,
                        help='the root path of datasets(must have subfolders: trainA, trainB, testA, testB')
    parser.add_argument('--weight_file', type=str, default='./weights', help='where to store weight file')
    parser.add_argument('--weight_from', type=str, default=None, help='where to load the pretrained weight')
    parser.add_argument('--A_nc', type=int, default=3, help='the number of A domain images\' channels')
    parser.add_argument('--B_nc', type=int, default=3, help='the number of B domain images\' channels')
    parser.add_argument('--res_layers', type=int, default=9, help='the number of resnet blocks in CycleGAN model')
    parser.add_argument('--init_type', type=str, default='normal', help='the initialization method')
    parser.add_argument('--init_gain', type=float, default=0.02, help='the scaling factor of initialization')
    parser.add_argument('--direction', type=str, default='A2B', help='A2B / B2A')
    parser.add_argument('--batch_size', type=int, default=1, help='the input batch size')
    parser.add_argument('--num_parallel', type=int, default=1, help='the num of workers in parallel to read data')
    parser.add_argument('--shuffle', type=bool, default=False, help='if True then shuffle the input images')
    parser.add_argument('--crop_size', type=int, default=None, help='the crop size')
    parser.add_argument('--crop_pos', type=tuple, default=None, help='the crop position')
    parser.add_argument('--flip', type=bool, default=None, help='if True then flip input images')
    parser.add_argument('--grayscale', type=bool, default=False, help='if True then read images as gray ones')
    parser.add_argument('--resize', type=int, default=None, help='the size of images to be resized')
    parser.add_argument('--convert', type=bool, default=True, help='if True then read images as torch.Tensor')
    parser.add_argument('--lr', type=float, default=0.002, help='the learning rate')
    parser.add_argument('--lambda_A', type=float, default=10, help='the factor of Cycle loss for A')
    parser.add_argument('--lambda_B', type=float, default=10, help='the factor of Cycle loss for B')
    parser.add_argument('--lambda_idt', type=float, default=0, help='the factor of identity loss')
    parser.add_argument('--gpu', type=bool, default=False, help='if True then use GPU')
    parser.add_argument('--epochs', type=int, default=1, help='the training epochs')

    args = parser.parse_args()

    params = TransParams(resize_osize=args.resize,
                         crop_size=args.crop_size,
                         crop_pos=args.crop_pos,
                         flip=args.flip)
    transformer = Transform(params.get_params(),
                            grayscale=args.grayscale,
                            resize=(args.resize is not None),
                            crop=(args.crop_size is not None),
                            flip=(args.flip is not None),
                            convert=args.convert)
    dataset = UnalignedDataset(root_path=args.dataroot, transformer=transformer, phase='train')
    dataloader = Dataloader(dataset=dataset, batch_sz=args.batch_size, shuffle=args.shuffle,
                            num_works=args.num_parallel)
    model = CycleGAN(A_nc=args.A_nc,
                     B_nc=args.B_nc,
                     res_block_num=args.res_layers,
                     init_type=args.init_type,
                     init_gain=args.init_gain,
                     lr=args.lr,
                     lambda_A=args.lambda_A,
                     lambda_B=args.lambda_B,
                     lambda_idt=args.lambda_idt)
    if args.weight_from is not None:
        model.load_state_dict(torch.load(args.weight_from))

    if args.gpu:
        model = model.cuda()

    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader):
            real_A = data['A']
            real_B = data['B']
            if args.gpu:
                real_A = real_A.cuda()
                real_B = real_B.cuda()

            model.set_input(real_A, real_B)
            model.optimize()
            model.print_loss_info(epoch)
        if epoch % 5 == 0 and epoch != 0:
            mkdir(args.weight_file)
            weight_file_path = os.path.join(args.weight_file, 'epoch_%d_cyclegan_weight.pth' % epoch)
            torch.save(model.state_dict(), weight_file_path)
