from __future__ import print_function, absolute_import

import os
import torch
import functools
import torch.utils.data as data
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class VideoDataset(data.Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self,
                 dataset_name,
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        # appearance
        img_paths, pid, camid = self.dataset[index]

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        # gait
        img_paths_gait = []
        if self.dataset_name == 'ilidsvid':
            for img_path in img_paths:
                path = img_path.replace('iLIDS-VID', 'iLIDS-VID-ECCV-mask')
                img_paths_gait.append(path)
        elif self.dataset_name == 'mars':
            for img_path in img_paths:
                path = img_path.replace('MARS', 'MARS-ECCV-mask')
                path = path.replace('.jpg', '.png')
                img_paths_gait.append(path)
        elif self.dataset_name == 'duke':
            for img_path in img_paths:
                path = img_path.replace('DukeMTMC-VideoReID', 'DukeMTMC-VideoReID-ECCV-mask')
                path = path.replace('.jpg', '.png')
                img_paths_gait.append(path)

        # if self.temporal_transform is not None:
        #     img_paths_gait = self.temporal_transform(img_paths_gait)  # random select 4 frames

        clip_gait = self.loader(img_paths_gait)

        if self.spatial_transform is not None:
            clip_gait = [self.spatial_transform(img) for img in clip_gait]

        clip_gait = torch.stack(clip_gait, 0).permute(1, 0, 2, 3)

        return clip, clip_gait, pid, camid


class ImageDataset(data.Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        transform (callable, optional): A function/transform that  takes in the
            imgs and transforms it.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, pid, camid) where pid is identity of the clip.
        """
        img_path, pid, camid = self.dataset[index]

        img = image_loader(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid