from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def getImage(dataset_path):
    '''
    获取当前目录下的所有pic_type格式的图片
    '''
    pics = sorted(glob(dataset_path+'/*.npy'))
    pic_data = []
    pic_name = []
    length_path = len(
        dataset_path)+1 if pics[0][len(dataset_path)] == '/' else len(dataset_path)
    # 或者 pics_name=[var.split('/')[-1] for var in pics]

    for pic in pics:
        pic_ = np.load(pic)
        pic_data.append(pic_)
        pic_name.append(pic[length_path:-4])

    return pic_data, pic_name


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
        self.pic_data, self.imgs_name = getImage(self.dataset_path)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.pic_data[idx]).unsqueeze(0), self.imgs_name[idx]

    def __len__(self):
        return len(self.imgs_name)


if __name__ == '__main__':
    test_data_loader = DataLoader(
        dataset=infer_DataSet('/home/e201cv/Desktop/covid_data/process_clf/test/imgs'), batch_size=4,
        num_workers=8, shuffle=False, drop_last=False)
    print(len(test_data_loader))
    for idx, (data, name) in enumerate(test_data_loader):
        print(name)
