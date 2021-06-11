import argparse
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def cal_class(masks_dir, out_dir):
    masks_files = glob(masks_dir+'/*.png')
    output_files = glob(out_dir+'/*.png')
    masks_data = []
    out_data = []
    for pic in masks_files:
        pic_ = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        masks_data.append(pic_)
    for pic in output_files:
        pic_ = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        out_data.append(pic_)
    out_data = np.array(out_data)
    out_data = np.where(out_data > 0, 1, 0)

    masks_data = np.array(masks_data)
    masks_data = np.where(masks_data > 1, 1, 0)
    cm = confusion_matrix(masks_data.ravel(), out_data.ravel())  # 支持任意类
    # 等同
    # pos_arr=np.zeros(shape=(3,3))
    # pos_arr[2,2]=np.where(((masks_data==2)&(out_data==2)),1,0).sum()   # 行为真实 列为判断
    # pos_arr[2,1]=np.where(((masks_data==2)&(out_data==1)),1,0).sum()
    # pos_arr[2,0]=np.where(((masks_data==2)&(out_data==0)),1,0).sum()
    # pos_arr[1,2]=np.where(((masks_data==1)&(out_data==2)),1,0).sum()
    # pos_arr[1,1]=np.where(((masks_data==1)&(out_data==1)),1,0).sum()
    # pos_arr[1,0]=np.where(((masks_data==1)&(out_data==0)),1,0).sum()
    # pos_arr[0,2]=np.where(((masks_data==0)&(out_data==2)),1,0).sum()
    # pos_arr[0,1]=np.where(((masks_data==0)&(out_data==1)),1,0).sum()
    # pos_arr[0,0]=np.where(((masks_data==0)&(out_data==0)),1,0).sum()
    # print(pos_arr)
    return cm


def main(args):
    out_dir = args.out_dir + \
        ('/'+args.model_name if not args.model_name in args.out_dir else '')
    print('out_dir:', out_dir)
    data = cal_class(masks_dir=args.masks_dir, out_dir=out_dir)
    print(data)
    tmp = data.sum(axis=1)  # axis=1每一行相加
    print_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        print_data[i, :] = np.array(data[i, :]/tmp[i])

    fig, ax = plt.subplots(figsize=(10, 10))
    conf_matrix = pd.DataFrame(
        print_data, index=['b&f', 'lesion'], columns=['b&f', 'lesion'])

    sns.heatmap(conf_matrix, fmt='g', square=True, annot=True,
                annot_kws={"size": 19}, cmap="Blues")
    plt.title(args.model_name+'_heatmap_each row equal to 1')
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if not os.path.exists('./output/heatmap/'):
        os.mkdir('./output/heatmap/')
    plt.savefig('./output/heatmap/'+args.model_name+'_2c.jpg')
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--masks_dir', type=str,
                         default='data/seg/test/masks')
    parser_.add_argument('--out_dir', type=str,
                         default='output/segResult')
    parser_.add_argument('--model_name', type=str, default='UNet')
    args = parser_.parse_args()
    main(args)
