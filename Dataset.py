from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import copy

class Dataset(Dataset):
    def __init__(self, name, train_split = .8, **kwargs) -> None:
        super().__init__()
        self.name = name
        self.train_split = train_split
        self.batch_size = kwargs.get("batch_size", 128)
        self.shuffle = kwargs.get("shuffle", True)
        self.num_workers = kwargs.get("num_workers", 1)
		
    def train(self) -> Dataset:
        raise NotImplementedError("The validation method should be implemented!")

    def validation(self) -> Dataset:
        raise NotImplementedError("The validation method should be implemented!")
    
    def dataloader(self, validation = False):
        tmp = self.validation() if validation else self.train()
        return DataLoader(tmp, batch_size = self.batch_size, shuffle=self.shuffle, num_workers = self.num_workers)
    
    def __getitem__(self, index):
        raise NotImplementedError("The validation method should be implemented!")
        
    def __len__(self):
        raise NotImplementedError("The validation method should be implemented!")
    
    def __iter__(self):
        for ix in range(0, len(self)):
            yield self[ix]
        