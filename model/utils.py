import torch
import torch.nn as nn
import random
import torch.nn.init as init


#############################################################
# This file defines some utility classes or functions used  #
# to help to build the training and testing system          #
#############################################################


class ImagePool(object):
    def __init__(self, pool_size=50):
        """
        create a image pool with pool_size

        :param pool_size: the size of image pool
        """
        self.pool_size = pool_size
        self.images = []

    def query(self, new_images):
        """
        update the image pool and return some images

        :param new_images: the new images used to update image pool randomly
        :return: some images
        """
        if self.pool_size == 0:
            return new_images
        ret = []
        for image in new_images:
            image = torch.unsqueeze(image.data, 0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                ret.append(image)

            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = image
                    ret.append(tmp)
                else:
                    ret.append(image)

        ret = torch.cat(ret, 0)
        return ret


class WeightInitializer(object):
    def __init__(self, init_type='normal', init_gain=0.02):
        assert init_type in ['normal', 'xavier', 'kaiming']
        self.init_type = init_type
        self.init_gain = init_gain

    def init(self, net):
        def init_func(n):
            classname = n.__class__.__name__
            if hasattr(n, 'weight') and (classname.find('conv') != -1 or classname.find('Linear') != -1):
                if self.init_type == 'normal':
                    init.normal_(n.weight.data, std=self.init_gain)
                elif self.init_type == 'xavier':
                    init.xavier_normal_(n.weight.data, gain=self.init_gain)
                elif self.init_type == 'kaiming':
                    init.kaiming_normal_(n.weight.data)
                else:
                    raise NotImplementedError('initialization method %s is not implemented' % self.init_type)
                if hasattr(n, 'bias') and n.bias is not None:
                    init.constant_(n.bias.data, 0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(n.weight.data, mean=1.0, std=self.init_gain)
                init.constant_(n.bias.data, 0)

        print("initialize network %s with %s method" % (net.__class__.__name__, self.init_type))
        net.apply(init_func)