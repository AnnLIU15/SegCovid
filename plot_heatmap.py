import argparse
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def cal_class(masks_dir, out_dir,num_classes=3):
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
    masks_data = np.array(masks_data)
    out_data = np.array(out_data)

    if num_classes==3:
        masks_data = np.where(masks_data > 1, masks_data-1, 0)
    elif num_classes==2:
        out_data = np.where(out_data > 0, 1, 0)
        masks_data = np.where(masks_data > 1, 1, 0)
    else:
        raise ValueError('unsupport num_classses')
    return confusion_matrix(masks_data.ravel(), out_data.ravel())  # 返回混淆矩阵


def main(args):
    out_dir = args.out_dir + \
        ('/'+args.model_name if not args.model_name in args.out_dir else '')
    print('out_dir:', out_dir)
    data = cal_class(masks_dir=args.masks_dir, out_dir=out_dir,num_classes=args.num_classes)
    print(data)
    tmp = data.sum(axis=0)  # axis=1每一行相加
    true_seg = np.zeros(data.shape)
    for i in range(data.shape[0]):
        true_seg[:, i] = np.array(data[:, i]/tmp[i])
    
    print(true_seg)
    label_disp=['bgr+lungs', 'GGO', 'CO'] if args.num_classes==3 else ['bgr+lungs', 'lesion']
    plt.subplots(figsize=(10, 10))
    conf_matrix = pd.DataFrame(
        true_seg, index=label_disp, columns=label_disp)

    sns.heatmap(conf_matrix, fmt='g', square=True, annot=True,
                annot_kws={"size": 19}, cmap="Blues")
    plt.title(args.model_name+'_heatmap')
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if not os.path.exists('./output/heatmap/'):
        os.mkdir('./output/heatmap/')
    plt.savefig('./output/heatmap/'+args.model_name+'_'+str(args.num_classes)+'c_.jpg')
    plt.close()
    

if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--masks_dir', type=str,
                         default='data/seg/test/masks')
    parser_.add_argument('--out_dir', type=str,
                         default='output/segResult')
    parser_.add_argument('--model_name', type=str, default='U2Net')
    parser_.add_argument('--num_classes', type=int, default=3)

    args = parser_.parse_args()
    main(args)
