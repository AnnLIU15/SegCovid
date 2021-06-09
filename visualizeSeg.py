# check preprocess by extract images from 3d numpy arrays

import os
from glob import glob
import argparse
from tqdm import tqdm
import matplotlib
import numpy as np
import cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main(args):
    # source data and label data
    imgsdir = args.imgsdir
    labeldir = args.labeldir
    outputdir = args.outputdir

    # save images
    save_visualize_dir = args.save_visualize_dir
    n_classes=args.n_classes
    multi=255//(n_classes-1)
    if not os.path.exists(save_visualize_dir):
        os.makedirs(save_visualize_dir)

    fig = plt.figure()
    imgx = fig.add_subplot(131)
    imgy = fig.add_subplot(132)
    imgz = fig.add_subplot(133)

    imgs=sorted(glob(imgsdir+"/*.jpg"))
    labels = sorted(glob(labeldir+"/*.png"))
    outputs= sorted(glob(outputdir+"/*.png"))

    for idx,label in tqdm(enumerate(labels),desc='processing',total=len(labels)):
        img=imgs[idx]
        output=outputs[idx]
        assert output[len(outputdir)+1:]==label[len(labeldir)+1:],"图片不匹配\t"+output[len(outputdir):]+'\t'+label[len(labeldir):]

        img_pic=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
        label_imgs = cv2.imread(label,cv2.IMREAD_GRAYSCALE)
        label_imgs=np.where( label_imgs>0,label_imgs-1,0)
        output_imgs = cv2.imread(output,cv2.IMREAD_GRAYSCALE)
        assert label_imgs.shape==output_imgs.shape,"图片维度不匹配"
        
        imgx.clear()
        imgy.clear()
        imgz.clear()
        imgz.imshow(img_pic, cmap='gray')
        imgz.set_title('imgs')
        imgx.imshow(label_imgs * multi, cmap='gray')
        imgx.set_title('label')
        imgy.imshow(output_imgs * multi, cmap='gray')
        imgy.set_title('output')

        imgfilename = label[len(labeldir)+1:-4]+ '.jpg'
        fig.set_size_inches(15, 8)

        plt.title(imgfilename)
        plt.savefig(os.path.join(save_visualize_dir, imgfilename))

if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--imgsdir', type=str,default='data/seg/test/imgs')
    parser_.add_argument('--labeldir', type=str,default='data/seg/test/masks')
    parser_.add_argument('--outputdir', type=str, default='output/segResult/U2Net')
    parser_.add_argument('--save_visualize_dir', type=str,default='output/segResult/U2Net_wn_visualize')
    parser_.add_argument('--n_classes', type=int, default=3)
    args = parser_.parse_args()

    main(args)
    

