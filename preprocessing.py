"""Original CFCD BGR center-crop input rule; augmentation is disabled."""
import numpy as np
import torch
from encoder_configuration import REPO
import datasets.transforms as official_transforms
from datasets.commondataset import _EIG_VALS,_EIG_VECS
MEAN=[0.406,0.456,0.485]
STD=[0.225,0.224,0.229]
class CenterCrop:
    image_size=512
    train_augment=False
    def _prepare(self, image):
        if self.train_augment:
            image = official_transforms.random_sized_crop(image, self.image_size, area_frac=0.08)
            image = official_transforms.horizontal_flip(image, 0.5, order="HWC")
        else:
            image = official_transforms.scale(self.image_size, image)
            image = official_transforms.center_crop(self.image_size, image)
        image = image.transpose([2, 0, 1]).astype(np.float32, copy=False) / 255.0
        if self.train_augment:
            image = official_transforms.lighting(image, 0.1, _EIG_VALS, _EIG_VECS)
        return torch.from_numpy(official_transforms.color_norm(image, MEAN, STD).copy())
