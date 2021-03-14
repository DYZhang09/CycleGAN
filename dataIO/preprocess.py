import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image


###########################################################
# This file defines some transforms used to transform     #
# the image (eg. crop, grayscale...)                      #
###########################################################


class Transform(object):
    def __init__(self, params, grayscale=False, resize=False, crop=False, flip=False, convert=False,
                 method=Image.BICUBIC):
        """
        initialize the image transformer

        :param params: the transformation params(must be the instance of dict)
        :param grayscale: if True then convert image to gray image
        :param resize: if True then resize image to params['output_size']
        :param crop: if True then crop image using params['crop_size'] and params['crop_pos']
        :param flip: if True then flip image using params['flip'] or randomly horizontal flip
        :param convert: if True then convert image to torch.Tensor and normalize to [-1, 1]
        :param method: interpolation method
        """
        self.transform_list = []
        assert isinstance(params, dict)

        if grayscale:
            self.transform_list.append(transforms.Grayscale())
        if resize:
            assert params['output_size'] is not None, "attempt to resize images without output_size"
            output_size = [params['output_size'], params['output_size']]
            self.transform_list.append(transforms.Resize(output_size, method))
        if crop:
            assert params['crop_size'] is not None, "attempt to crop images without crop_size"
            if params['crop_pos'] is None:
                self.transform_list.append(transforms.RandomCrop(params['crop_size']))
            else:
                self.transform_list.append(
                    transforms.Lambda(lambda img: self.crop(img, params['crop_pos'], params['crop_size'])))
        if flip:
            if params['flip'] is None:
                self.transform_list.append(transforms.RandomHorizontalFlip())
            else:
                self.transform_list.append(transforms.Lambda(lambda image: self.flip(image, params['flip'])))
        if convert:
            self.transform_list.append(transforms.ToTensor())
            if grayscale:
                self.transform_list.append(transforms.Normalize((0.5,), (0.5,)))
            else:
                self.transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        self.transform = transforms.Compose(self.transform_list)

    def crop(self, image, pos, size):
        ow, oh = image.size
        x, y = pos
        dw = dh = size
        if ow > dw or oh > dh:
            return image.crop(x, y, x + dw, y + dh)
        return image

    def flip(self, image, flip):
        if flip:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def apply(self, image):
        """
        apply the transformation on image

        :param image: input image
        :return: transformed image
        """
        return self.transform(image)


class TransParams(object):
    def __init__(self, resize_osize=None, crop_size=None, crop_pos=None, flip=None):
        """
        set params for image transformation

        :param resize_osize: the output size of resizing image
        :param crop_size: the output size of cropping image
        :param crop_pos: the position where to crop image
        :param flip: if True then flip image left and right
        """
        self.params = {'output_size': None, 'crop_size': None, 'crop_pos': None, 'flip': None}
        if resize_osize is not None:
            self.params['output_size'] = resize_osize
        if crop_size is not None:
            self.params['crop_size'] = crop_size
        if crop_pos is not None:
            self.params['crop_pos'] = crop_pos
        if flip is not None:
            self.params['flip'] = flip

    def get_params(self):
        """
        get the params(dict)

        :return: the transformation params
        """
        return self.params
