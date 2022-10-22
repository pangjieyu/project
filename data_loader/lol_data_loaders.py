import sys
sys.path.append('./data_loader')
sys.path.append('.')
from torchvision import datasets, transforms
from base import BaseDataLoader
from lol_dataset import LolDataset


class LolDataloader(BaseDataLoader):
    """
    LOL Dataloader
    """ 

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        trsfm = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

        self.data_dir = data_dir
        self.dataset = LolDataset(data_dir, trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)