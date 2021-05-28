from glob import glob
import os
import cv2
import numpy as np
import argparse


def PreImg(imgs_data):
    imgs_data_copy = sorted(imgs_data.copy().reshape(-1))
    imgs_data_shape = imgs_data.shape
    idx = 1
    for var in imgs_data_shape:
        idx *= var
    # https://www.zhihu.com/question/379900540/answer/1411664196
    idx_0_005, idx_0_995 = imgs_data_copy[round(
        0.005*idx)], imgs_data_copy[round(0.995*idx)]
    
    imgs_data = np.where(imgs_data < idx_0_005, idx_0_005, imgs_data)
    imgs_data = np.where(imgs_data > idx_0_995, idx_0_995, imgs_data)

    imgs_data = (imgs_data-imgs_data.mean())/imgs_data.std() # z-score
    return imgs_data


def PreMask(mask_data, n_classes=3):
    
    if n_classes == 2:
        output = np.where(mask_data > 1.5, 1, 0)
    elif n_classes == 3:
        output = np.where(mask_data > 1.5, mask_data-1, 0)
    
    return output


def getTotal(dataset_path, n_classes=3, normalize=True):
    '''
    获得dataset_path/imgs与dataset_path/masks下的图片
    '''
    imgs_path = dataset_path+'/imgs'
    mask_path = dataset_path+'/masks'
    print('='*30)
    print('process imgs')
    imgs_data, imgs_name = getImage(
        imgs_path, img_type='imgs', n_classes=n_classes, pic_type='.jpg', normalize=normalize)
    print('='*30)
    print('process masks')
    masks_data, masks_name = getImage(
        mask_path, img_type='masks', n_classes=n_classes, pic_type='.png', normalize=normalize)
    assert imgs_name == masks_name, '掩膜相片与相片对应不上'
    return imgs_data,masks_data, masks_name


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
        print(pic[length_path:],pic_.shape,pic_.max())
        pic_data.append(pic_)
        pic_name.append(pic[length_path:-4])
    return np.array(pic_data), pic_name

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
        print(pic[length_path:],pic_.shape,pic_.max())
        pic_data.append(pic_)
        pic_name.append(pic[length_path:-4])
    return np.array(pic_data), pic_name

def imgs_normalize(dataset_path, normalize=True):
    '''
    获取当前目录下的所有pic_type格式的图片
    '''
    pics = sorted(glob(dataset_path+'/*.png'))
    length_path = len(dataset_path)+1
    
    for pic in pics:
        pic_ = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        pic_=PreImg(pic_)
        print(pic[length_path:],pic_.shape,pic_.max())



def main(args):
    print('imgs dir:',args.in_dir)
    print('imgs save dir:',args.out_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        os.mkdir(args.out_dir+'/imgs')
    if not os.path.exists(args.out_dir+'/masks'+str(args.n_classes)):
        os.mkdir(args.out_dir+'/masks'+str(args.n_classes))
    imgs_data,masks_data, masks_name=getTotal(args.in_dir,n_classes=args.n_classes)
    for idx,_ in enumerate(masks_name):
        np.save(args.out_dir+'/imgs/'+masks_name[idx]+'.npy',imgs_data[idx])
        np.save(args.out_dir+'/masks'+str(args.n_classes)+'/'+masks_name[idx]+'.npy',masks_data[idx])
    print('total imgs(&masks):',len(masks_name))
if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--in_dir', type=str,
                         default='data/seg/train/')
    parser_.add_argument('--n_classes', type=int,
                         default=3)
    parser_.add_argument('--out_dir', type=str,
                         default='data/seg/process/train')
    args = parser_.parse_args()
    main(args)
    