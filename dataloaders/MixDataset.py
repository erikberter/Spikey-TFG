
import torch
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from dataloaders.dataset import VideoDataset

class MixedDataset(Dataset):
    def __init__(self, dataset = "hmdb51", split = "train", clip_len = 16):

        self.datasetA = VideoDataset(dataset=dataset, split=split, clip_len=clip_len, preprocess = False)
        self.datasetB = VideoDataset(dataset=dataset + "_flow", split=split, clip_len=clip_len, preprocess = False)
        
    def __getitem__(self, index):
        xA, label = self.datasetA[index]
        xB, _ = self.datasetB[index]
        
        return (xA, xB), label
    
    def __len__(self):
        return len(self.datasetA)
