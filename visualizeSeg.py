import argparse
import os
from glob import glob

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm



def main(args):
    # source data and label data
    imgsdir = args.imgsdir
    labeldir = args.labeldir
    outputdir = args.outputdir

    # save visualize images path
    save_visualize_dir = args.save_visualize_dir
    n_classes = args.n_classes
    # n类的灰度值(0为背景)
    multi = 255//(n_classes-1)
    if not os.path.exists(save_visualize_dir):
        os.makedirs(save_visualize_dir)
    # 从左到右子图依次为label predict_output imgs
    matplotlib.use('Agg')  # 图像不显示，需要在plt声明前使用
    import matplotlib.pyplot as plt

    fig = plt.figure()
    imgx = fig.add_subplot(131)
    imgy = fig.add_subplot(132)
    imgz = fig.add_subplot(133)
    # 获取这些图片，sorted保证数据对其
    imgs = sorted(glob(imgsdir+"/*.jpg"))
    labels = sorted(glob(labeldir+"/*.png"))
    outputs = sorted(glob(outputdir+"/*.png"))

    for idx, label in tqdm(enumerate(labels), desc='processing', total=len(labels)):
        img = imgs[idx]
        output = outputs[idx]
        # 判断是否对其，必要时可以改为try except
        assert output[len(outputdir)+1:] == label[len(labeldir)+1:], "图片不匹配\t" + \
            output[len(outputdir):]+'\t'+label[len(labeldir):]
        # 读取
        img_pic = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        label_imgs = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        output_imgs = cv2.imread(output, cv2.IMREAD_GRAYSCALE)
        assert label_imgs.shape == output_imgs.shape, "图片维度不匹配"

        # 将肺野和背景归为一类
        label_imgs = np.where(label_imgs > 0, label_imgs-1, 0)

        # 清除以前图像信息
        imgx.clear()
        imgy.clear()
        imgz.clear()
        # 显示
        imgx.imshow(label_imgs * multi, cmap='gray')
        imgx.set_title('label')
        imgy.imshow(output_imgs * multi, cmap='gray')
        imgy.set_title('output')
        imgz.imshow(img_pic, cmap='gray')
        imgz.set_title('imgs')
        fig.set_size_inches(15, 8)
        # 保存图片
        imgfilename = label[len(labeldir)+1:-4] + '.jpg'
        plt.title(imgfilename)
        plt.savefig(os.path.join(save_visualize_dir, imgfilename))


if __name__ == '__main__':
    r'''
    本函数为可视化函数，需要在segTest或infer运行后得到掩膜相片才可使用
    ----------------------------------------------------------
    |Input:
    |args->     imgsdir     labeldir     outputdir      save_visualize_dir      n_classes
    |           图像位置      掩膜位置      预测图保存位置   可视化保存位置             确定显示种类
    |Output:
    |visualize_pic
    '''
    # 加载参数
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--imgsdir', type=str, default='data/seg/test/imgs')
    parser_.add_argument('--labeldir', type=str, default='data/seg/test/masks')
    parser_.add_argument('--outputdir', type=str,
                         default='output/segResult/U2Net')
    parser_.add_argument('--save_visualize_dir', type=str,
                         default='output/segResult/U2Net_wn_visualize')
    parser_.add_argument('--n_classes', type=int, default=3)
    # 获取参数
    args = parser_.parse_args()
    # visualize
    main(args)
