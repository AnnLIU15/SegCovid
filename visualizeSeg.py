# check preprocess by extract images from 3d numpy arrays

import os
from glob import glob

import matplotlib
import numpy as np
import cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# source data and label data
imgsdir = 'data/seg/test/imgs'
labeldir = 'data/seg/test/masks'
outputdir = 'output/segResult/U2Net'

# save images of left and right chests
check_dir = 'output/segResult/visualize'
if not os.path.exists(check_dir):
    os.makedirs(check_dir)

fig = plt.figure()
imgx = fig.add_subplot(131)
imgy = fig.add_subplot(132)
imgz = fig.add_subplot(133)

imgs=sorted(glob(imgsdir+"/*.jpg"))
labels = sorted(glob(labeldir+"/*.png"))
outputs= sorted(glob(outputdir+"/*.png"))

for idx,label in enumerate(labels):
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
    imgx.imshow(label_imgs * 127, cmap='gray')
    imgx.set_title('label')
    imgy.imshow(output_imgs * 127, cmap='gray')
    imgy.set_title('output')

    imgfilename = label[len(labeldir)+1:-4]+ '.jpg'
    fig.set_size_inches(15, 8)

    plt.title(imgfilename)
    plt.savefig(os.path.join(check_dir, imgfilename))

