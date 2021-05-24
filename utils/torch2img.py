import cv2
import torch
import numpy as np
def torch2imgs(output,imgs_type='.png'):
    imgs_array=output.argmax(dim=1)


if __name__ == '__main__':
    # imgs=torch.randint(0,256,(512,512))
    # cv2.imwrite('./output/lena.png',imgs.numpy(), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])#第三个参数表示的是压缩级别。默认为3.
    # cv2.imwrite('./output/1lena.png',imgs.numpy(), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])#第三个参数表示的是压缩级别。默认为3.取最小就好
    imgs1=cv2.imread('./output/lena.png',cv2.IMREAD_GRAYSCALE)
    imgs=cv2.imread('./output/1lena.png',cv2.IMREAD_GRAYSCALE)
    print(imgs1==imgs,(imgs1-imgs).sum())