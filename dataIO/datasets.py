import os
import random
import torch
import torch.nn as nn
from PIL import Image
import torch.utils.data as data
from dataIO.utils import *
from dataIO.preprocess import *


###########################################################
# This file defines the dataset classes used to read data #
# from files or write data into files.                    #
###########################################################


# ---------------------------- dataset --------------------------------------#
class UnalignedDataset(data.Dataset):
    def __init__(self, root_path, transformer=None, phase='train'):
        """
        initialize the unaligned dataset that stores unpaired data.

        +root_path\n
        |-----+trainA\n
        |-----+trainB\n
        |-----+testA\n
        |-----+testB\n

        :param root_path: the root path of dataset.
        :param transformer: the transformer used to apply image transformation(must be the instance of class Transform)
        :param phase: 'train' for training, 'test' for test
        """
        super(UnalignedDataset, self).__init__()
        self.root_path = root_path
        self.phase = phase
        self.A_paths = []
        self.B_paths = []
        self.A_paths = mk_img_paths(os.path.join(root_path, phase + 'A'))
        self.B_paths = mk_img_paths(os.path.join(root_path, phase + 'B'))
        if transformer is not None:
            assert isinstance(transformer, Transform)
            self.transformer = transformer

    def __len__(self):
        return max(len(self.A_paths), len(self.B_paths))

    def __getitem__(self, item):
        A_path = self.A_paths[item % len(self.A_paths)]
        # use random to avoid pairing data implicitly
        B_path = self.B_paths[random.randint(0, len(self.B_paths) - 1)]
        img_A = img_read(A_path)
        img_B = img_read(B_path)
        if self.transformer is not None:
            img_A = self.transformer.apply(img_A)
            img_B = self.transformer.apply(img_B)
        return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}


# ---------------------------- dataloader -------------------------------------#
class Dataloader(object):
    def __init__(self, dataset, batch_sz, shuffle=True, num_works=1):
        """
        initialize the dataloader.

        :param dataset: the dataset(must be the instance of torch.utils.data.Dataset)
        :param batch_sz: batch size
        :param shuffle: if True then shuffle data
        :param num_works: the num of workers working in parallel
        """
        assert isinstance(dataset, data.Dataset), "the dataset is not a instance of torch.utils.data.Dataset"
        self.dataset = dataset
        self.dataloader = data.DataLoader(dataset, batch_sz, shuffle, num_workers=num_works)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


# ----------------------------- imgwriter ------------------------------------#
class Imgwriter(object):
    def __init__(self, file_root):
        """
        initialize the image writer.

        :param file_root: the root of path
        """
        self.root = file_root

    def write(self, img_tensor, img_paths, domain='A'):
        """
        write images into the paths

        :param img_tensor: the images' torch.Tensor
        :param img_paths: the paths where images will reside
        :param domain: which domain does image belong to
        :return: None
        """
        assert len(img_paths) == img_tensor.shape[0], "size of paths and num of images not match!"
        path = os.path.join(self.root, domain)
        if not os.path.exists(path):
            mkdir(path)
        for i in range(len(img_paths)):
            img = tensor2npimg(img_tensor[i], scale=True)
            _, filename = os.path.split(img_paths[i])
            img_write(img, os.path.join(path, filename))


# briefly test the implementation above
if __name__ == '__main__':
    transparams = TransParams(resize_osize=128, crop_size=64)
    transformer = Transform(params=transparams.get_params(), resize=True, flip=True, convert=True, crop=True, grayscale=True)
    dataset = UnalignedDataset(r'../data/horse2zebra', phase='test', transformer=transformer)
    dataloader = Dataloader(dataset, batch_sz=50, num_works=4, shuffle=False)
    writer = Imgwriter(r'../data/horse2zebra/fake')
    for i, data in enumerate(dataloader):
        A, B, A_paths, B_paths = data['A'], data['B'], data['A_path'], data['B_path']
        writer.write(A, fake_img_paths(A_paths))
        writer.write(B, fake_img_paths(B_paths))
