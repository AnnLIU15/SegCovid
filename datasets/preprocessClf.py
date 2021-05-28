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


def imgs_normalize(in_dir,out_dir):
    '''
    获取当前目录下的所有pic_type格式的图片
    '''
    pics = sorted(glob(in_dir+'/*.png'))
    length_path = len(in_dir)+1

    print('imgs dir:',in_dir)
    print('imgs save dir:',out_dir)
    print('total imgs:',len(pics))
    for pic in pics:
        pic_ = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        pic_=PreImg(pic_)
        np.save(out_dir+'/imgs/'+pic[length_path:-4]+'.npy',pic_)
    