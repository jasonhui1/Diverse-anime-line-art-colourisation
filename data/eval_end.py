from __future__ import division
import math
import os
import os.path
import random
import numbers
import numpy as np
# from scipy.misc import fromimage
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import Resize, CenterCrop
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.9, 1.) * area
            aspect_ratio = random.uniform(7. / 8, 8. / 7)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Resize((self.size,self.size), interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root, path):
    images = []

    for root, __, fnames in sorted(os.walk(os.path.join(root, path))):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)

                images.append(path)
    return images

def color_loader(path):
    return Image.open(path).convert('RGB')



def sketch_loader(path):
    return Image.open(path).convert('L')


def resize_by(img, img2, side_min):
    return img.resize((int(img.size[0] / min(img.size) * side_min),
                       int(img.size[1] / min(img.size) * side_min)),
                      Image.BICUBIC), \
           img2.resize((int(img.size[0] / min(img.size) * side_min),
                       int(img.size[1] / min(img.size) * side_min)),
                      Image.BICUBIC), 


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1):
        w, h = img1.size
        th, tw = self.size
        if w == tw and h == th:  # ValueError: empty range for randrange() (0,0, 0)
            return img1

        if w == tw:
            x1 = 0
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th))

        elif h == th:
            x1 = random.randint(0, w - tw)
            y1 = 0
            return img1.crop((x1, y1, x1 + tw, y1 + th))

        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th))

def resize_by(img, side_min):
    return img.resize((int(img.size[0] / min(img.size) * side_min),
                       int(img.size[1] / min(img.size) * side_min)),
                      Image.BICUBIC)

class ImageFolder(data.Dataset):
    def __init__(self, root, stransform=None):
        # imgs = make_dataset(root)
        #sketch_imgs = sorted(make_dataset(root, 'Sketch/authentic_line_art_256'))
        sketch_imgs_nc = sorted(make_dataset(root, 'Sketch/authentic_line_art_noclean_256'))
        if len(sketch_imgs_nc) == 0:
            raise (RuntimeError("Found 0 images in folders."))
        self.root = root
        #self.imgs = sketch_imgs
        self.imgs_nc = sketch_imgs_nc
        self.stransform = stransform

    def __getitem__(self, index):

        #fname = self.imgs[index]  # random.randint(1, 3
        fname_nc = self.imgs_nc[index]  # random.randint(1, 3

        #print(fname)
        print(fname_nc)

        #Simg = sketch_loader(fname)
        Simg_nc = sketch_loader(fname_nc)
        Simg_nc = resize_by(Simg_nc, 256.0)


        Simg_nc = RandomCrop(256)(Simg_nc)

        if random.random() < 0.5:
            Simg_nc= Simg_nc.transpose(Image.FLIP_LEFT_RIGHT)
        #Simg = self.stransform(Simg)
        Simg_nc = self.stransform(Simg_nc)

        return Simg_nc

    def __len__(self):
        return len(self.imgs_nc)


def CreateDataLoader(config):
    random.seed(config.seed)

    STrans = transforms.Compose([
        transforms.Resize((config.image_size,config.image_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ])


    dataset = ImageFolder(root=config.val_end_root, stransform=STrans)

    assert dataset

    return data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=1, drop_last=False)