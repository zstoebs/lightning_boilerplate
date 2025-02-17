from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import numpy as np

class ABCDataset(ABC, Dataset):
    def __init__(self): 
        super().__init__()
    
    @abstractmethod
    def change_stage(self): 
        pass