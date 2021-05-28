import torch.nn.functional as F
import torch
import os
import numpy as np



def one_hot_mask(data,n_classes=3):
    '''
    对掩膜数据进行one_hot编码
    请先把data转为灰度值模式
    data.shape=(batch_size,1,x,y)

    output.shape=(batch_size,n_classes,x,y)
    '''
    shape_of_data=np.array(data.shape)
    shape_of_data[1]=n_classes
    output=torch.zeros(size=tuple(shape_of_data))
    if shape_of_data[0]==1:
        output=one_hot_single(data,n_classes)
    else:
        for i in range(shape_of_data[0]):
            output[i]=one_hot_single(data[i],n_classes)
    return output
def one_hot_single(data,n_classes=3):
    '''
    对当张图片进行one_hot编码
    data.shape=(1,x,y)
    1->灰度值模式
    n_classes->输出output.shape=(n_classes,x,y)
    目前只支持两类和三类，更多请自行修改
    '''
    '''
    data=0=>背景
    data=1=>肺野
    data=2=>GGO肺磨玻璃影
    data=3->CO肺实质
    
    n_classes=2=>   背景+肺野=0
                    GGO+CO=1

    n_classes=3=>   背景+肺野=0
                    GGO=1
                    CO=2               
    '''
    return F.one_hot(data,n_classes).permute(0,3,1,2)
