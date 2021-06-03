from glob import glob
import torch
import numpy as np
from torch.utils.data import Dataset

class infer_DataSet(Dataset):
    '''
    CNCB数据库
    dataset_path为数据路径
    目录结构为
    >dataset_path
        >imgs
            >xxxx.jpg
        >masks
            >xxxx.png
    '''

    def __init__(self, dataset_path):
        super(infer_DataSet, self).__init__()
        self.dataset_path = dataset_path
        self.imgs_name=sorted(glob(self.dataset_path+'/*.npy'))

    def __getitem__(self, idx):
        return torch.FloatTensor(np.load(self.imgs_name[idx])).unsqueeze(0), self.imgs_name[idx][len(self.dataset_path):]

    def __len__(self):
        return len(self.imgs_name)



