# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2020-12-08 23:48:30
# @Last Modified by:   ZhaoYang
# @Last Modified time: 2021-06-02 17:12:33

import argparse
import os
import shutil
from glob import glob


class BatchRenamePics(object):

    def __init__(self, path, save_path, train_txt, test_txt):
        super(BatchRenamePics, self).__init__()
        # 设置起始路径path
        self.path = path
        self.save_path = save_path
        self.train_txt = train_txt
        self.test_txt = test_txt

    def _divide(self):
        folder = [self.save_path+'/'+var for var in os.listdir(
            self.save_path) if ('mask' in var) or ('img' in var)]
        to_path = [self.save_path+'/train', self.save_path+'/test']
        for var in to_path:
            if not os.path.exists(var):
                os.mkdir(var)
                os.mkdir(var+'/imgs')
                os.mkdir(var+'/masks')
        with open(self.train_txt, 'r') as f:
            train_pic = sorted([var.replace('\n', '')
                               for var in f.readlines()])
        with open(self.test_txt, 'r') as f:
            test_pic = sorted([var.replace('\n', '') for var in f.readlines()])
        for idx, fold in enumerate(folder):
            print('data_dir:', fold)
            if 'imgs' in fold:
                type_pic = 'imgs'
            elif 'masks' in fold:
                type_pic = 'masks'

            files = sorted(glob(fold+'/*'))
            print('train save dir:', to_path[0]+'/'+type_pic)

            for pic_ in train_pic:
                for file in files:
                    if pic_[:-4] == file[len(fold)+1:-4]:
                        print(file, '->', to_path[0]+'/'+type_pic)

                        shutil.move(file, to_path[0]+'/'+type_pic)
                        continue
            print('test save dir:', to_path[1]+'/'+type_pic)
            for pic_ in test_pic:
                for file in files:
                    if pic_[:-4] == file[len(fold)+1:-4]:
                        print(file, '->', to_path[1]+'/'+type_pic)

                        shutil.move(file, to_path[1]+'/'+type_pic)
                        # test_pic.remove(pic_)
                        continue

    def rename_divide_preprocess(self):
        folder = [self.path+'/'+var for var in os.listdir(
            self.path) if ('mask' in var) or ('image' in var)]
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for type_img in folder:
            if 'image' in type_img:
                save_dir = self.save_path+'/imgs'
            elif 'mask' in type_img:
                save_dir = self.save_path+'/masks'
            else:
                continue
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            i = 0
            for root, _, files in os.walk(type_img):
                if i < 2:
                    i += 1
                    continue
                name_list = [(root+'/'+file) for file in files]
                for var in name_list:
                    ''' 
                    var[len(self.path)+1:][::-1].replace('/','_',1)[::-1] 解释
                    >>> var
                    >>> /home/e201cv/Desktop/dataset/nCoVR/ct_lesion_seg/mask/19/84.png
                    >>> var[len(self.path)+1:]
                    >>> mask/19/84.png
                    >>> var[len(self.path)+1:][::-1]
                    >>> gnp.48/91/ksam
                    >>> var[len(self.path)+1:][::-1].replace('/','_',1)将前1个/换为_
                    >>> gnp.48_91/ksam
                    >>> var[len(self.path)+1:][::-1].replace('/','_',1)[::-1]
                    >>> mask/19_84.png
                    '''
                    shutil.copyfile(
                        var, save_dir+'/'+var[len(type_img)+1:][::-1].replace('/', '_', 1)[::-1])
        self._divide()


if __name__ == '__main__':
    # 设置起始路径path
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--path', '-p', type=str,
                         default='/home/e201cv/Desktop/dataset/nCoVR/ct_lesion_seg', help='unzip ct_lesion_seg.zip path')
    parser_.add_argument('--save_path', '-s', type=str,
                         default='/home/e201cv/Desktop/dataset/nCoVR/mytest', help='save rename pic')
    parser_.add_argument('--train_txt', type=str,
                         default='./data/train_split_segmentation.txt', help='save rename pic')
    parser_.add_argument('--text_txt', type=str,
                         default='./data/test_split_segmentation.txt', help='save rename pic')
    args = parser_.parse_args()
    # 创建实例对象
    pics = BatchRenamePics(args.path, args.save_path,
                           args.train_txt, args.text_txt)
    # 调用实例方法
    pics.rename_divide_preprocess()
