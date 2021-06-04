import argparse
from glob import glob
from multiprocessing import Process
import os

import cv2
import numpy as np
import SimpleITK as sitk
from numpy.core.fromnumeric import ptp, shape
from radiomics import featureextractor


def multi_process_extract_radiomics(imgs_path, masks_path, save_path):
    if isinstance(imgs_path, str):
        imgs_path = [imgs_path]
    if isinstance(masks_path, str):
        masks_path = [masks_path]
    if isinstance(save_path, str):
        save_path = [save_path]
    assert len(imgs_path) == len(
        masks_path), "imgs_path's num not equal to masks_path's"
    assert len(imgs_path) == len(
        save_path), "imgs_path's num not equal to save_path's"
    assert len(masks_path) == len(
        save_path), "masks_path's num not equal to save_path's"
    process_list = []
    for idx, _ in enumerate(imgs_path):
        if not os.path.exists(save_path[idx]):
            os.mkdir(save_path[idx])
        process_list.append(Process(target=extract_radiomics, args=(
            imgs_path[idx], masks_path[idx], save_path[idx],)))
    for process_ in process_list:
        process_.start()


def extract_radiomics(imgs_path, masks_path, save_path):
    r'''
    注意这里的imgs需要经过z-score处理后的标准化numpy.ndarray
    masks图片需要先处理
    '''
    img_files = sorted(glob(imgs_path+'/*.npy'))
    img_names = [var[len(imgs_path):-4] for var in img_files]
    masks_files = sorted(glob(masks_path+'/*.npy'))
    masks_names = [var[len(masks_path):-4] for var in masks_files]
    assert img_names == masks_names, '图像不对应'
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    kernel=np.ones(shape=(3,3),dtype=np.uint8)
    print('imgs_path:',imgs_path,'masks_path:',masks_path,
        '\ntotal pic_:',len(img_names),'save_path:',save_path)
    for idx, img_name in enumerate(img_names):
        save_name=save_path+img_name+'.npy'
        if os.path.exists(save_name):
            continue
        np_array_radiomics = np.zeros(shape=(1032),dtype=np.float64)
        
        
        masks = np.load(masks_files[idx]).squeeze()

        masks = np.where(masks > 0.5, 1, 0)
        # masks_process = masks

       
        masks_process=cv2.morphologyEx(
            src=cv2.morphologyEx(src=masks.astype(np.float32),op=cv2.MORPH_OPEN,kernel=kernel),
            op=cv2.MORPH_CLOSE,kernel=kernel
            )
        '''
        by the open and close operation, we filled tiny holes in objects and 
        Eliminate small objects, separate objects in the thin and smooth boundaries of larger objects
        '''
        if not masks_process.sum() == 0:
            '''
            if it hasn't mask dont process it
            '''
            imgs = np.load(img_files[idx])
            sitk_img = sitk.GetImageFromArray(imgs)
            sitk_img.SetSpacing((1, 1, 1))

            sitk_mask = sitk.GetImageFromArray(masks_process)
            print(img_name,masks.shape,masks.sum())
            sitk_mask.SetSpacing((1, 1, 1))
            result = extractor.execute(sitk_img, sitk_mask)
            for idx, (_, var) in enumerate(result.items()):
                if idx > 21:
                    np_array_radiomics[idx-22]=var
        ## 保存
        np.save(save_name,np_array_radiomics)
            #print(np_array_radiomics)
            # with open('aaa.txt','w+') as f:
            #     for idx, (key, var) in enumerate(result.items()):
            #         if idx > 21:
            #             print('idx:',idx,'key:',key,'var',var,type(var),var.shape,file=f)
            #             np_array_radiomics[idx-22]=var
            # print(np_array_radiomics.dtype)



def main(args):
    multi_process_extract_radiomics(args.imgs_dir, args.masks_dir, args.out_dir,)


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--imgs_dir', type=str, nargs='+',
                         default='data/clf/train/imgs')
    parser_.add_argument('--masks_dir', type=str, nargs='+',
                         default='data/clf/train/masks/')
    parser_.add_argument('--out_dir', type=str, nargs='+',
                         default='data/clf/train/radiomics')
    args = parser_.parse_args()
    main(args)
