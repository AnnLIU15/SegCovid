import os

import cv2
import numpy as np


def torch2imgs(output):
    '''
    函数为将torch数据转为2维图片保存
    '''
    shapeOFoutput = output.shape
    if len(shapeOFoutput) == 3 and shapeOFoutput[0] > 1:
        # 将one-hot结果选取最大值
        imgs_array = output.argmax(dim=0).clone().detach().cpu().numpy()
    elif len(shapeOFoutput) == 3 and shapeOFoutput[0] == 1:
        # 去掉无用维度
        imgs_array = output.clone().detach().cpu().numpy()
    elif len(shapeOFoutput) == 4:
        # batchsize~=1，递归保存(反正不考虑效率)
        imgs_array = np.zeros(
            shape=(shapeOFoutput[0], shapeOFoutput[2], shapeOFoutput[3]))
        for idx in range(shapeOFoutput[0]):
            imgs_array[idx] = torch2imgs(output[idx])
    else:
        assert False, '2D图片的输出维度应该是4.维度不匹配,目前维度'+str(len(shapeOFoutput))
    return imgs_array


def saveImage(imgs_array, name_of_imgs, save_dir='./output/segResult/', npy_type=False):
    '''
    使用OPENCV保存图片
    '''
    # 转为ndarray
    get_numpy = torch2imgs(imgs_array).astype(np.uint8)

    shapeofimg = get_numpy.shape
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # 保存
    if len(shapeofimg) == 3:
        for idx in range(shapeofimg[0]):
            if isinstance(name_of_imgs[idx], tuple):
                writer_name = name_of_imgs[idx][0]
            else:
                writer_name = name_of_imgs[idx]
            if npy_type:
                writer_name = writer_name+'.png'
            cv2.imwrite(save_dir+'/'+writer_name, get_numpy[idx], [
                        int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    else:
        if isinstance(name_of_imgs, tuple):
            name_of_imgs = name_of_imgs[0]
        if npy_type:
            writer_name = name_of_imgs+'.png'
        cv2.imwrite(save_dir+'/'+writer_name, get_numpy,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    # import torch
    # imgs=torch.randint(0,256,(512,512))
    # cv2.imwrite('./output/lena.png',imgs.numpy(), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])#第三个参数表示的是压缩级别。默认为3.
    # cv2.imwrite('./output/1lena.png',imgs.numpy(), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])#第三个参数表示的是压缩级别。默认为3.取最小就好
    imgs1 = cv2.imread('./output/lena.png', cv2.IMREAD_GRAYSCALE)
    imgs = cv2.imread('./output/1lena.png', cv2.IMREAD_GRAYSCALE)
    print(imgs1 == imgs, (imgs1-imgs).sum())
