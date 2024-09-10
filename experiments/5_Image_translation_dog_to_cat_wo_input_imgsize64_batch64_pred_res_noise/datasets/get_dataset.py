import argparse

from datasets.base import Dataset
from datasets.generation import get_dataset


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def dataset(folder,
            image_size,
            exts=['jpg', 'jpeg', 'png', 'tiff'],
            augment_flip=False,
            convert_image_to=None,
            condition=0,
            equalizeHist=False,
            crop_patch=True,
            sample=False, 
            generation=False):
    if generation:
        # dataset_import = "generation"
        # dataset = "CELEBA"
        # args = {"exp": "xxx/dataset/diffusion_dataset"}

        dataset_import = "base"
    else:
        dataset_import = "base"

    if dataset_import == "base":
        return Dataset(folder,
                       image_size,
                       exts=exts,
                       augment_flip=augment_flip,
                       convert_image_to=convert_image_to,
                       condition=condition,
                       equalizeHist=equalizeHist,
                       crop_patch=crop_patch,
                       sample=sample)
    elif dataset_import == "generation":
        if dataset == "CELEBA":
            config = {
                "data": {
                    "dataset": "CELEBA",
                    "image_size": 64,  # 64
                    "channels": 3,
                    "logit_transform": False,
                    "uniform_dequantization": False,
                    "gaussian_dequantization": False,
                    "random_flip": True,
                    "rescaled": True,
                }}
        elif dataset == "CIFAR10":
            config = {
                "data": {
                    "dataset": "CIFAR10",
                    "image_size": 32,  # 32
                    "channels": 3,
                    "logit_transform": False,
                    "uniform_dequantization": False,
                    "gaussian_dequantization": False,
                    "random_flip": True,
                    "rescaled": True,
                }}
        elif dataset == "bedroom":
            config = {
                "data": {
                    "dataset": "LSUN",
                    "category": "bedroom",
                    "image_size": 256,  # 256
                    "channels": 3,
                    "logit_transform": False,
                    "uniform_dequantization": False,
                    "gaussian_dequantization": False,
                    "random_flip": True,
                    "rescaled": True,
                }}
        elif dataset == "church_outdoor":
            config = {
                "data": {
                    "dataset": "LSUN",
                    "category": "church_outdoor",
                    "image_size": 256,  # 256
                    "channels": 3,
                    "logit_transform": False,
                    "uniform_dequantization": False,
                    "gaussian_dequantization": False,
                    "random_flip": True,
                    "rescaled": True
                }}
        args = dict2namespace(args)
        config = dict2namespace(config)
        return get_dataset(args, config)[0]
