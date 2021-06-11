import argparse
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


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
    imgs_data = np.array(imgs_data, dtype=np.uint8)
    # imgs_data = (imgs_data-imgs_data.mean())/imgs_data.std()  # z-score
    # 注意z-score会使得数据从uint8转为float64 1B->8B，内存不够慎用
    # 但是z-score可以提升效果
    return imgs_data


def imgs_normalize(in_dir, out_dir):
    '''
    获取当前目录下的所有pic_type格式的图片
    '''
    pics = sorted(glob(in_dir+'/*.png'))
    length_path = len(in_dir)+1
    if not os.path.exists(out_dir+'/imgs/'):
        os.makedirs(out_dir+'/imgs/')
    print('imgs dir:', in_dir)
    print('imgs save dir:', out_dir)
    len_pic = len(pics)
    print('total imgs:', len_pic)
    for pic in tqdm(pics, desc='Process', total=len_pic):
        pic_ = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        pic_ = PreImg(pic_)
        # print(out_dir+'/imgs/'+pic[length_path:-4]+'.npy')
        np.save(out_dir+'/imgs/'+pic[length_path:-4]+'.npy', pic_)


def main(args):
    in_dir = args.in_dir
    out_dir = args.out_dir
    if isinstance(in_dir, str):
        in_dir = [in_dir]
    if isinstance(out_dir, str):
        out_dir = [out_dir]
    assert len(in_dir) == len(
        out_dir), "in_dir's number is not equal to out_dir's"
    for idx, _ in enumerate(in_dir):
        if not os.path.exists(out_dir[idx]):
            os.makedirs(out_dir[idx])
        imgs_normalize(in_dir[idx], out_dir[idx])


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--in_dir', type=str, nargs='+',
                         default='/home/e201cv/Desktop/covid_data/clf/train')
    parser_.add_argument('--out_dir', type=str, nargs='+',
                         default='/home/e201cv/Desktop/covid_data/process_clf/train')
    args = parser_.parse_args()
    main(args)
