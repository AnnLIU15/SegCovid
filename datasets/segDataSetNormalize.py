from glob import glob

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np



def getTotal(dataset_path, n_classes=3):
    '''
    获得dataset_path/imgs与dataset_path/masks下的图片
    '''
    imgs_path = dataset_path+'/imgs'
    mask_path = dataset_path+'/masks'+str(n_classes)
    imgs_data, imgs_name = getImage(
        imgs_path)
    masks_data, masks_name = getImage(
        mask_path)
    assert imgs_name == masks_name, '掩膜相片与相片对应不上'
    return torch.FloatTensor(imgs_data).unsqueeze(dim=1), torch.LongTensor(masks_data), masks_name


def getImage(dataset_path):
    '''
    获取当前目录下的所有pic_type格式的图片
    '''
    pics = sorted(glob(dataset_path+'/*.npy'))
    pic_data = []
    pic_name = []
    length_path = len(dataset_path)+1
    
    for pic in pics:
        pic_ = np.load(pic)
        pic_data.append(pic_)
        pic_name.append(pic[length_path:-4])
    
    return pic_data, pic_name


class COVID19_SegDataSetNormalize(Dataset):
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

    def __init__(self, dataset_path, n_classes=3):
        super(COVID19_SegDataSetNormalize, self).__init__()
        self.dataset_path = dataset_path
        self.imgs_data, self.masks_data, self.imgs_name = getTotal(
            self.dataset_path,n_classes)

    def __getitem__(self, idx):
        return self.imgs_data[idx], self.masks_data[idx]

    def __len__(self):
        return len(self.imgs_name)


class COVID19_SegDataSetNormalize_test(Dataset):
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

    def __init__(self, dataset_path, n_classes=3):
        super(COVID19_SegDataSetNormalize_test, self).__init__()
        self.dataset_path = dataset_path
        self.imgs_data, self.masks_data, self.imgs_name = getTotal(
            self.dataset_path, n_classes)

    def __getitem__(self, idx):
        return self.imgs_data[idx], self.masks_data[idx], self.imgs_name[idx]

    def __len__(self):
        return len(self.imgs_name)


if __name__ == '__main__':
    dataset = COVID19_SegDataSetNormalize('data/seg/process/test')
    data_loader = DataLoader(
        dataset=dataset, batch_size=8, num_workers=8, shuffle=False, drop_last=False)
    for batch_idx, (data, target) in enumerate(data_loader):
        print(batch_idx, data.shape, target.shape,
              target[0].shape, target[0].unique())
        # tmp_a=one_hot_mask(target[0],3)
        # tmp_a1=one_hot_mask(target[1],3)
        # tmp_a2=one_hot_mask(target[2],3)
        # tmp_b=one_hot_mask(target,3)
        # print(torch.sum(tmp_b[0]-tmp_a[0]))
        # print(torch.sum(tmp_b[1]-tmp_a1[0]))
        # print(torch.sum(tmp_b[2]-tmp_a2[0]))
    # imgs_data,masks_data,_=getTotal('data/seg/test')
    # masks_data=torch.LongTensor(masks_data)
    # masks_data=masks_data.unsqueeze(dim=1)
    # n_classes=3
    # for var in masks_data:
    #     var=torch.where(var>1.0,var-1,0)
    #     tmp=one_hot_single(var,n_classes)
    #     #tmp=torch.nn.functional.one_hot(var,num_classes=n_classes).permute(0,3,1,2)
    #     # for n in range(n_classes-1,0,-1):
    #     #     if torch.unique(var)[1:].shape[0]<n:
    #     #         tmp=torch.cat((tmp,torch.zeros_like(var.unsqueeze(dim=0))),dim=1)

    #     print(tmp.shape,torch.unique(var),tmp[:,2,:,:].sum())
