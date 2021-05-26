import seaborn as sns
import numpy as np
import  matplotlib.pyplot as plt
import pylab as mpl     #import matplotlib as mpl
from glob import glob
 
def main():
    #设置汉字格式
    # sans-serif就是无衬线字体，是一种通用字体族。
    # 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica,SimHei 中文的幼圆、隶书等等
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    data =np.array([[9014,424,12],[643,6625,127],[44,476,3826]])
    #np.array([[8872,570,8],[539,6758,98],[63,412,3871]])
    #np.array([[95.39,4.49,0.13],[8.7,89.59,1.72],[1.01,10.95,88.03]])
    sns.heatmap(data,fmt='g', square=True,annot=True,cmap="Blues",yticklabels=['新型冠状肺炎','肺炎','正常'],xticklabels=['新型冠状肺炎','肺炎','正常'])
    # plt.title('3分类问题混淆矩阵')
    plt.title('三分类问题混淆矩阵')


    plt.show()

if __name__ == '__main__':
    main()
     
    