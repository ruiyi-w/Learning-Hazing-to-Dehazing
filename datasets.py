import os
import cv2
import glob
import torch
import random
import numpy as np
import torch.utils.data as data

from torchvision.transforms import ToTensor
from utils.data_utils import random_crop, add_JPEG_noise, synthesize


class RealHazyData(data.Dataset):
    def __init__(self, image_folder, crop_size=512):
        super().__init__()
        self.crop_size = crop_size
        self.image_folder = image_folder
        self.image_names = [
            os.path.basename(name)
            for name in glob.glob(os.path.join(image_folder, '*.jpg'))
        ]

    def __getitem__(self, index):
        name = self.image_names[index]
        real_hazy = cv2.imread(self.image_folder + name)
        real_hazy = cv2.cvtColor(real_hazy, cv2.COLOR_BGR2RGB)
        hazy = ToTensor()(random_crop(real_hazy, self.crop_size)) * 2. - 1.
        return hazy.clip(-1., 1.), "hazy, foggy, misty, obscure, smoggy."

    def __len__(self):
        return len(self.image_names)


class StaticPairedData(data.Dataset):
    def __init__(self, hazy_folder, clean_folder, crop_size=448):
        super().__init__()
        self.hazy_folder = hazy_folder
        self.clean_folder = clean_folder
        self.hazy_names = [
            os.path.basename(name)
            for name in glob.glob(os.path.join(hazy_folder, '*.png'))
        ]
        self.crop_size = crop_size

    def __len__(self):
        return len(self.hazy_names)

    def __getitem__(self, index):
        hazy_name = self.hazy_names[index]
        clean = cv2.imread(os.path.join(self.clean_folder, hazy_name.split('_')[0] + '.jpg'))
        hazy = cv2.imread(os.path.join(self.hazy_folder, hazy_name))
        stack = np.concatenate((clean, hazy), axis=2)
        stack = random_crop(stack, self.crop_size)
        clean, hazy = stack[:, :, :3], stack[:, :, 3:]

        clean = clean.astype(np.float32) / 255.0
        hazy = hazy.astype(np.float32) / 255.0

        if np.random.rand() < 0.5:
            hazy = add_JPEG_noise(hazy)

        clean = ToTensor()(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        hazy = ToTensor()(cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB))

        clean = (clean * 2.0 - 1.0).clip(-1.0, 1.0)
        hazy = hazy.clip(0.0, 1.0)
        return clean, hazy, "sharp, clear-sky, vivid, well-defined, brightly lit, unobstructed, detailed, balanced composition, neutral-toned, natural contrast, evenly exposed, modern aesthetic, minimal atmospheric distortion, realistic yet polished look, high visual clarity, refreshing openness"


class HybridTrainingData(data.Dataset):
    def __init__(self, syn_folder, real_folder, crop_size=448, p=0.3):
        super().__init__()
        syn_names = [os.path.basename(name) for name in glob.glob(os.path.join(syn_folder, 'rgb_500/*.jpg'))]
        real_names = [os.path.basename(name) for name in glob.glob(os.path.join(real_folder, '*.jpg'))]

        self.p = p
        self.syn_folder = syn_folder
        self.real_folder = real_folder
        self.syn_names = syn_names
        self.real_names = real_names
        self.crop_size = crop_size

    def __getitem__(self, index):
        if np.random.rand(1) < self.p:
            syn_clean_name = self.syn_names[index]
            syn_clean = cv2.imread(self.syn_folder + 'rgb_500/' + syn_clean_name).astype(np.float32) / 255.0
            depth = np.load(os.path.join(self.syn_folder + 'depth_500/', syn_clean_name.split('.')[0] + '.npy'))

            syn_hazy = synthesize(img_gt=syn_clean, img_depth=depth).astype(np.float32) / 255.0
            syn_clean = cv2.cvtColor(syn_clean, cv2.COLOR_BGR2RGB)

            stack = np.concatenate((syn_clean, syn_hazy), axis=2)
            stack = random_crop(stack, self.crop_size)
            clean, hazy = stack[:, :, :3], stack[:, :, 3:]

            clean = ToTensor()(clean).clip(0., 1.)
            hazy = ToTensor()(hazy).clip(0., 1.)

            # [-1, 1] for target, [0, 1] for condition
            return hazy * 2. - 1., clean, "hazy, foggy, misty, obscure, smoggy.", 'cond'

        else:
            name = random.choice(self.real_names)
            real_hazy = cv2.imread(self.real_folder + name)
            real_hazy = cv2.cvtColor(real_hazy, cv2.COLOR_BGR2RGB)

            real_hazy = ToTensor()(random_crop(real_hazy, self.crop_size)) * 2. - 1.
            return real_hazy.clip(-1., 1.), torch.zeros_like(real_hazy, dtype=real_hazy.dtype), "hazy, foggy, misty, obscure, smoggy.", 'uncond'

    def __len__(self):
        return len(self.syn_names)
