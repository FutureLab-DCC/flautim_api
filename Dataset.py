from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose


class Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
		
    def train(self) -> DataLoader:
        pass

    def validation(self) -> DataLoader:
        pass        