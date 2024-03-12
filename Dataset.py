from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose


class Dataset(Dataset):
    def __init__(self, name, **kwargs) -> None:
        super().__init__()
        self.name = name
		
    def train(self) -> DataLoader:
        raise NotImplementedError("The train method should be implemented!")

    def validation(self) -> DataLoader:
        raise NotImplementedError("The validation method should be implemented!")