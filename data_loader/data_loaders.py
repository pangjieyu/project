from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.lol_dataset import LolDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        trsfm = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class LolDataloader(BaseDataLoader):
    """
    LOL Dataloader
    """ 

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        trsfm = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.ToTensor()
        ])

        self.data_dir = data_dir
        self.dataset = LolDataset(data_dir, trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class TestDataloader(BaseDataLoader):
    """
    LOL Dataloader
    """ 

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])

        self.data_dir = data_dir
        self.dataset = LolDataset(data_dir, trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)