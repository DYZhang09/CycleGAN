import os
import torch
from PIL import Image
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_img(filename):
    """
    if a filename indicates that the file is an image
    """
    return any(filename.endswith(ext) for ext in IMG_EXTENSIONS)


def mk_img_paths(file_root, max_size=float("inf")):
    """
    do a path walk to find all images(path) under the root folder

    :param file_root: the root folder path
    :param max_size: the max size of images to find
    :return: all the image paths under the file_root
    """
    assert os.path.isdir(file_root)

    image_paths = []
    for root, _, filenames in sorted(os.walk(file_root)):
        for filename in filenames:
            if is_img(filename):
                image_paths.append(os.path.join(root, filename))
    return image_paths[:min(max_size, len(image_paths))]


def mkdir(dir):
    """
    make a directory recursively
    """
    if not os.path.exists(dir):
        os.makedirs(dir, 0o777)


def img_read(path, numpy=False):
    """
    read an image from path

    :param path: where the image resides
    :param numpy: if True then convert image to numpy.ndarray
    :return: the image
    """
    assert is_img(path)

    img = Image.open(path).convert('RGB')
    if numpy:
        return np.array(img)
    return img


def img_write(image_np, path):
    """
    write an image into path

    :param image_np: the numpy array of image
    :param path: the path to write into
    :return: None
    """
    img = Image.fromarray(image_np)
    img.save(path)


def tensor2npimg(img_tensor, scale=True, img_type=np.uint8):
    """
    transform image tensor to numpy images

    :param img_tensor: the tensor of images
    :param scale: if True then scale the tensor([-1, 1]) to [0, 255]
    :param img_type: the type of numpy array
    :return: the numpy array of images
    """
    if not isinstance(img_tensor, np.ndarray):
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.data
        else:
            return img_tensor

        img_np = img_np.cpu().float().numpy()
        if img_np.shape[0] == 1:
            img_np = np.tile(img_np, (3, 1, 1))
        if scale:
            img_np = (np.transpose(img_np, (1, 2, 0)) + 1) / 2.0 * 255.0
        else:
            img_np = np.transpose(img_np, (1, 2, 0))
        return img_np.astype(img_type)
    else:
        return img_tensor.astype(img_type)


def fake_img_paths(real_img_paths):
    """
    modify image paths to get the corresponding fake image paths by inserting '_fake' before
    the extension name

    :param real_img_paths: the paths of real images
    :return: the corresponding paths of fake images
    """
    fake_paths = []
    if not isinstance(real_img_paths, list):
        real_img_paths = [real_img_paths]
    for path in real_img_paths:
        for ext in IMG_EXTENSIONS:
            if path.endswith(ext):
                fake_paths.append(path.replace(ext, '_fake' + ext))
    return fake_paths
