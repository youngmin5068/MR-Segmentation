from torch.utils.data import DataLoader
from config import *

def data_load(dataset,batch_size=BATCH_SIZE,train=True,shuffle=True):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=12,pin_memory=True)
    size = len(dataset)
    
    return data_loader, size

