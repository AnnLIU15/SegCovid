from glob import glob

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


def PreImg(imgs_data):
    if isinstance(imgs_data, np.ndarray):
        imgs_data = torch.from_numpy(imgs_data)
    imgs_data_copy = sorted(imgs_data.clone().reshape(-1))
    imgs_data_shape = imgs_data.shape
    idx = 1
    for var in imgs_data_shape:
        idx *= var
    # https://www.zhihu.com/question/379900540/answer/1411664196
    idx_0_005, idx_0_995 = imgs_data_copy[round(
        0.005*idx)], imgs_data_copy[round(0.995*idx)]
    imgs_data = torch.where(imgs_data < idx_0_005, idx_0_005, imgs_data)
    imgs_data = torch.where(imgs_data > idx_0_995, idx_0_995, imgs_data).float()
    del imgs_data_copy
    imgs_data = (imgs_data-imgs_data.mean())/imgs_data.std()
    return imgs_data


def PreMask(mask_data, n_classes=3):
    if isinstance(mask_data, np.ndarray):
        mask_data = torch.from_numpy(mask_data)
    shapeOfMask = mask_data.shape
    if len(shapeOfMask) == 3:
        if n_classes == 2:
            output = torch.where(mask_data > 1.5, 1, 0)
        elif n_classes == 3:
            output = torch.where(mask_data > 1.5, mask_data-1, 0)
    elif len(shapeOfMask) == 4:
        output = torch.zeros_like(shapeOfMask)
        for idx in range(shapeOfMask[0]):
            output[idx] = PreMask(mask_data[0], n_classes)
    return output


def getTotal(dataset_path, n_classes=3, normalize=True):
    '''
    获得dataset_path/imgs与dataset_path/masks下的图片
    '''
    imgs_path = dataset_path+'/imgs'
    mask_path = dataset_path+'/masks'
    imgs_data, imgs_name = getImage(
        imgs_path, img_type='imgs', n_classes=n_classes, pic_type='.jpg', normalize=normalize)
    masks_data, masks_name = getImage(
        mask_path, img_type='masks', n_classes=n_classes, pic_type='.png', normalize=normalize)
    tmp_masks_name = [var.replace('png', 'jpg') for var in masks_name]
    assert imgs_name[:-4] == tmp_masks_name[:-4], '掩膜相片与相片对应不上'
    print(imgs_data.shape,masks_data.shape)
    exit(1)
    return imgs_data.unsqueeze(dim=1), \
        PreMask(torch.LongTensor(masks_data), n_classes=n_classes), masks_name


def getImage(dataset_path, img_type, n_classes, pic_type='.jpg', normalize=True):
    '''
    获取当前目录下的所有pic_type格式的图片
    '''
    pics = sorted(glob(dataset_path+'/*'+pic_type))
    pic_data = []
    pic_name = []
    length_path = len(dataset_path)+1
    
    for pic in pics:
        pic_ = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        if img_type=='imgs' and normalize:
            pic_=PreImg(pic_)
        elif img_type=='masks':
            pic_=PreMask(pic_,n_classes=n_classes)
        else:
            raise RuntimeError('unknow picture type')
        print(pic_.shape)
        pic_data.append(pic_)
        pic_name.append(pic[length_path:])
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
            self.dataset_path)

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

    def __init__(self, dataset_path, n_classes=3, normalize=True):
        super(COVID19_SegDataSetNormalize_test, self).__init__()
        self.dataset_path = dataset_path
        self.imgs_data, self.masks_data, self.imgs_name = getTotal(
            self.dataset_path, n_classes, normalize)

    def __getitem__(self, idx):
        return self.imgs_data[idx], self.masks_data[idx], self.imgs_name[idx]

    def __len__(self):
        return len(self.imgs_name)


if __name__ == '__main__':
    dataset = COVID19_SegDataSetNormalize('data/seg/test')
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
