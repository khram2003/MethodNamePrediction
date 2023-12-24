
from typing import List
from torch.utils.data import Dataset
from dataset.utils import Method


class MethodNameDataset(Dataset):

    def __init__(self, methods: List[Method]):
        self.methods = methods


    def __len__(self):
        return len(self.methods)

    def __getitem__(self, idx):
        return self.methods[idx].body, self.methods[idx].name
