from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import torch
import copy
import numpy as np
import uuid

def collate_fn_antigo(data):
    imgs = []
    lbls = []
    for img, lbl in data:
        imgs.append(np.array(img, dtype=np.float32))  
        lbls.append(np.array(lbl, dtype=np.int64))    
    imgs = torch.from_numpy(np.stack(imgs, axis=0)) 
    lbls = torch.tensor(lbls, dtype=torch.int64)     
    return imgs, lbls


def collate_fn(data):
    imgs = []
    lbls = []
    for img, lbl in data:
        imgs.append(torch.tensor(np.array(img, dtype=np.float32)))  # Convert each image directly to a PyTorch tensor
        lbls.append(torch.tensor(lbl, dtype=torch.int64))           # Convert labels to PyTorch tensor as well
    imgs = torch.stack(imgs, dim=0)  # Stack tensors along a new dimension
    lbls = torch.tensor(lbls, dtype=torch.int64)  # Convert list of labels to tensor
    return imgs, lbls

class Dataset(Dataset):
    def __init__(self, name, train_split = .8, **kwargs) -> None:
        super(Dataset, self).__init__()
        self.id = kwargs.get("id",str(uuid.uuid1()))
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
        #return DataLoader(tmp, batch_size = self.batch_size, pin_memory=False, shuffle=self.shuffle, num_workers = self.num_workers, collate_fn=collate_fn)
        return DataLoader(tmp, batch_size = self.batch_size, num_workers = 1, collate_fn=collate_fn)
    
    def __getitem__(self, index):
        raise NotImplementedError("The validation method should be implemented!")
        
    def __len__(self):
        raise NotImplementedError("The validation method should be implemented!")
    
    def __iter__(self):
        for ix in range(0, len(self)):
            yield self[ix]
        
