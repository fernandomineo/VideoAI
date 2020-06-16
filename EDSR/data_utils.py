import pathlib
import os
from shutil import copyfile

from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import random

def getpathsx(path):
    """
    Get all image paths from folder 'path'.
    """
    data = pathlib.Path(path)
    all_image_paths = list(data.glob('*'))
    all_image_paths = [str(p) for p in all_image_paths]
    return all_image_paths

def getpaths(path):
    """
    Get all image paths from folder 'path' while avoiding ._ files.
    """
    im_paths = []
    for fil in os.listdir(path):
            if '.png' in fil:
                if "._" in fil:
                    #avoid dot underscore
                    pass
                else:
                    im_paths.append(os.path.join(path, fil))
    return im_paths

def make_val_dataset(paths, scale, mean):
    """
    Python generator-style dataset for the validation set. Creates input and ground truth.
    """
    for p in paths:
        # normalize
        im_norm = cv2.imread(p.decode(), 3).astype(np.float32) - mean

        # divisible by scale - create low-res
        hr = im_norm[0:(im_norm.shape[0] - (im_norm.shape[0] % scale)),
                  0:(im_norm.shape[1] - (im_norm.shape[1] % scale)), :]
        lr = cv2.resize(hr, (int(hr.shape[1] / scale), int(hr.shape[0] / scale)),
                        interpolation=cv2.INTER_CUBIC)

        yield lr, hr

def make_dataset(paths, scale, mean):
    """
    Python generator-style dataset. Creates 48x48 low-res and corresponding high-res patches.
    """
    size_lr = 48
    size_hr = size_lr * scale

    for p in paths:
        # normalize
        im_norm = cv2.imread(p.decode(), 3).astype(np.float32) - mean

        # random flip
        r = random.randint(-1, 2)
        if not r == 2:
            im_norm = cv2.flip(im_norm, r)

        # divisible by scale - create low-res
        hr = im_norm[0:(im_norm.shape[0] - (im_norm.shape[0] % scale)),
                  0:(im_norm.shape[1] - (im_norm.shape[1] % scale)), :]
        lr = cv2.resize(hr, (int(hr.shape[1] / scale), int(hr.shape[0] / scale)),
                        interpolation=cv2.INTER_CUBIC)

        numx = int(lr.shape[0] / size_lr)
        numy = int(lr.shape[1] / size_lr)

        for i in range(0, numx):
            startx = i * size_lr
            endx = (i * size_lr) + size_lr

            startx_hr = i * size_hr
            endx_hr = (i * size_hr) + size_hr

            for j in range(0, numy):
                starty = j * size_lr
                endy = (j * size_lr) + size_lr
                starty_hr = j * size_hr
                endy_hr = (j * size_hr) + size_hr

                crop_lr = lr[startx:endx, starty:endy]
                crop_hr = hr[startx_hr:endx_hr, starty_hr:endy_hr]

                x = crop_lr.reshape((size_lr, size_lr, 3))
                y = crop_hr.reshape((size_hr, size_hr, 3))

                yield x, y

def calcmean(imageFolder, bgr):
    """
    Calculates the mean of a dataset.
    """
    paths = getpaths(imageFolder)

    total_mean = [0, 0, 0]
    im_counter = 0

    for p in paths:

        image = np.asarray(Image.open(p))

        mean_rgb = np.mean(image, axis=(0, 1), dtype=np.float64)

        if im_counter % 50 == 0:
            print("Total mean: {} | current mean: {}".format(total_mean, mean_rgb))

        total_mean += mean_rgb
        im_counter += 1

    total_mean /= im_counter

    # rgb to bgr
    if bgr is True:
        total_mean = total_mean[...,::-1]

    return total_mean


def img_train_test_split(img_source_dir, train_size):
    """
    Randomly splits images over a train and validation folder, while preserving the folder structure

    Parameters
    ----------
    img_source_dir : string
        Path to the folder with the images to be split. Can be absolute or relative path

    train_size : float
        Proportion of the original images that need to be copied in the subdirectory in the train folder
    """
    if not (isinstance(img_source_dir, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(img_source_dir):
        raise OSError('img_source_dir does not exist')

    if not (isinstance(train_size, float)):
        raise AttributeError('train_size must be a float')

    # Set up empty folder structure if not exists
    if not os.path.exists('data'):
        os.makedirs('data')
    else:
        if not os.path.exists(os.path.join('data', 'train')):
            os.makedirs(os.path.join('data', 'train'))
        if not os.path.exists(os.path.join('data', 'validation')):
            os.makedirs(os.path.join('data', 'validation'))

    # Get the subdirectories in the main image folder
    #subdirs = [subdir for subdir in os.listdir(img_source_dir) if os.path.isdir(os.path.join(img_source_dir, subdir))]

    #for subdir in subdirs:
        #subdir_fullpath = os.path.join(img_source_dir, subdir)
        #if len(os.listdir(subdir_fullpath)) == 0:
        #    print(subdir_fullpath + ' is empty')
        #    break

    train_subdir = os.path.join('data', 'train')
    validation_subdir = os.path.join('data', 'validation')

        # Create subdirectories in train and validation folders
    #    if not os.path.exists(train_subdir):
    #        os.makedirs(train_subdir)

    #   if not os.path.exists(validation_subdir):
    #        os.makedirs(validation_subdir)

    train_counter = 0
    validation_counter = 0
    for filename in os.listdir(img_source_dir):
        ## Randomly assign an image to train or validation folder
        #for filename in os.listdir(subdir_fullpath):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            fileparts = filename.split('.')

            if random.uniform(0, 1) <= train_size:
                copyfile(os.path.join(img_source_dir, filename),
                             os.path.join(train_subdir,
                                          str(random.randint(0, 5000000)) + "_" + str(train_counter) + '.' +
                                          fileparts[1]))
                train_counter += 1
            else:
                copyfile(os.path.join(img_source_dir, filename),
                             os.path.join(validation_subdir,
                                          str(random.randint(0, 500000)) + "_" + str(validation_counter) + '.' +
                                          fileparts[1]))
                validation_counter += 1
#
    print('Copied ' + str(train_counter) + ' images to data/train/')
    print('Copied ' + str(validation_counter) + ' images to data/validation/')


#print(calcmean('./../train/', False))
#img_train_test_split('images/', 0.7)