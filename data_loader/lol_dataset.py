import os
from glob import glob

import torch
from PIL import Image
from torchvision import datasets, transforms


class LolDataset(torch.utils.data.Dataset):  # type: ignore
    """
    LolDataset
    """

    def __init__(self, data_dir, transform, train=True):
        super(LolDataset, self).__init__()
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.low_files = glob(self.data_dir + r'low/*.png')
        self.low_files += glob(self.data_dir + r'low/*.jpg')

    def __getitem__(self, index):
        if self.train:
            file = self.low_files[index]
            file_name = os.path.basename(file)
            high_file = os.path.join(self.data_dir + r'high/', file_name)
            low_img = self.transform(Image.open(file))
            high_img = self.transform(Image.open(high_file))

            return low_img, high_img
        return

    def __len__(self):
        return len(self.low_files)
