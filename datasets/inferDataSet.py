from glob import glob

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
        return np.load(self.imgs_name[idx]), self.imgs_name[idx][len(self.dataset_path)+1:]

    def __len__(self):
        return len(self.imgs_name)



