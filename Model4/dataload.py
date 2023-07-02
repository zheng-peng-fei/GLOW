import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_file):
        self.data = np.load(data_file)
        self.x_0 = torch.from_numpy(self.data['arr_0'])
        self.x_T = torch.from_numpy(self.data['arr_1'])
    
    def __getitem__(self, idx):
        x_0 = self.x_0[idx]
        x_T = self.x_T[idx]
        return x_0, x_T
    
    def __len__(self):
        return len(self.x_0)


#data_file = 'mydata.npz'
#batch_size = 32

#my_dataset = MyDataset(data_file)
#my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
