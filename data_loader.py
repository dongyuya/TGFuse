# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
from scipy.misc import imread, imsave, imresize
from torchvision import transforms, utils
from PIL import Image
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import matplotlib as mpl

#==========================dataset load==========================
def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images

def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)

def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)

# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_image(path, height=256, width=256):
    image = Image.open(path).convert('RGB')

    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')
    return image


def get_train_images_auto(paths, height=256, width=256):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width,)
        image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

# def get_test_images(path_con,path_sty, height=None, width=None):
#     ImageToTensor = transforms.Compose([transforms.ToTensor()])
#
#     cons = []
#     stys = []
#
#     con = get_image(path_con, height, width)
#     sty = get_image(path_sty, height, width)
#     w_con, h_con = con.size
#     w_sty, h_sty = sty.size
#     w = w_con if w_con<w_sty else w_sty
#     h = h_con if h_con<h_sty else h_sty
#     w = int(w / 32) * 32
#     h = int(h / 32) * 32
#
#     con = con.resize((w, h))
#     con = ImageToTensor(con).float().numpy()*255
#     cons.append(con)
#     cons = np.stack(cons, axis=0)
#     cons = torch.from_numpy(cons).float()
#
#     sty = sty.resize((w, h))
#     sty = ImageToTensor(sty).float().numpy() * 255
#     stys.append(sty)
#     stys = np.stack(stys, axis=0)
#     stys = torch.from_numpy(stys).float()
#
#     return cons, stys

def get_test_images(paths, height=None, width=None):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width)
        w, d = image.size
        w = int(w / 32) * 32
        d = int(d / 32) * 32
        image = image.resize((w, d))
        image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


# colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)


def save_images(path, data, img):
    w, d = img.size
    # data = data.reshape([data.shape[0], data.shape[1]])
    data = imresize(data, [d, w], interp='nearest')
    imsave(path, data)