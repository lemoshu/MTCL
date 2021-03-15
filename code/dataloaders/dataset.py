import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler

# Adapt to IRCAD SSL setting
class BaseDataSets_IRCAD_SSL(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slice_SSL_c.txt', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.txt', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        # Change the organ here
        organ ='ROI'
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/training_slice_{}_SSL_h5/{}.h5".format(organ, case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/val_{}_h5/{}.h5".format(organ, case), 'r')
        image = h5f['image'][:]
        label = h5f['label_{}'.format(organ)][:]
        sample = {'image': image, 'label_{}'.format(organ): label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample

# Adapt to IRCAD
class BaseDataSets_IRCAD(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slice.txt', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.txt', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        ######################## Change the organ here
        organ ='ROIori'
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/training_slice_{}_h5/{}.h5".format(organ, case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/val_{}_h5/{}.h5".format(organ, case), 'r')
        image = h5f['image'][:]
        label = h5f['label_{}'.format(organ)][:]
        sample = {'image': image, 'label_{}'.format(organ): label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class RandomGenerator_IRCAD(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        ########################## change the organ
        organ = 'ROIori'
        image, label = sample['image'], sample['label_{}'.format(organ)]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label_{}'.format(organ): label}
        return sample




"""
The Following Code is for concatenated input (2-channel)
"""

# Adapt to concat IRCAD
class BaseDataSets_IRCAD_concat(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slice.txt', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.txt', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        ######################## Change the organ here
        organ ='ROI'
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/training_slice_{}_concat_h5/{}.h5".format(organ, case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/val_{}_concat_h5/{}.h5".format(organ, case), 'r')
        image = h5f['image'][:]
        label = h5f['label_{}'.format(organ)][:]
        sample = {'image': image, 'label_{}'.format(organ): label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


# Adapt to concat IRCAD Mix
class BaseDataSets_IRCAD_Mix_concat(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slice_SSL_concat.txt', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.txt', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        ######################## Change the organ here
        organ ='ROI'
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/training_slice_{}_concat_SSL_h5/{}.h5".format(organ, case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/val_{}_concat_h5/{}.h5".format(organ, case), 'r')
        image = h5f['image'][:]
        label = h5f['label_{}'.format(organ)][:]
        sample = {'image': image, 'label_{}'.format(organ): label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


# Adapt to SSL-based concatenated IRCAD
class BaseDataSets_IRCAD_SSL_concat(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slice_SSL_concat.txt', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.txt', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        # Change the organ here
        organ ='ROI'
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/training_slice_{}_concat_SSL_h5/{}.h5".format(organ, case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/val_{}_concat_h5/{}.h5".format(organ, case), 'r')

        image = h5f['image'][:]
        label = h5f['label_{}'.format(organ)][:]
        sample = {'image': image, 'label_{}'.format(organ): label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


class RandomGenerator_IRCAD_concat(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        ########################## change the organ
        organ = 'ROI'
        image, label = sample['image'], sample['label_{}'.format(organ)]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        image = torch.from_numpy(image.astype(np.float32)) # remove the unsqueeze operation as we already have 2-channel
        # print(image.shape)
        label = torch.from_numpy(label.astype(np.uint8))
        # print(label.shape)
        sample = {'image': image, 'label_{}'.format(organ): label}
        return sample


# -----------------------------------------------------
class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
