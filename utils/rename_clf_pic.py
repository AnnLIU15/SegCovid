# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2020-12-08 23:48:30
# @Last Modified by:   ZhaoYang
# @Last Modified time: 2021-06-13 16:03:30

import os


class BatchRenamePics(object):
    '''
    批量重命名clf图片-> 种类_患者ID_扫描ID_slicerNum.png/jpg
    种类 0->Normal 1->CP 2-NCP
    '''
    def __init__(self, path):
        # 设置起始路径path
        self.path = path

    def rename(self):
        allfile = os.walk(self.path)
        # j用于计数,统计有多少张照片被重命名
        j = 0
        # 遍历每一层目录,从上到下的顺序
        for dirpath, dirnames, filenames, in allfile:
            # 得到当前文件夹的名字tail
            tail = os.path.split(dirpath)[1]
            # i用于命名
            i = 0
            # 遍历filenames中的每一个文件
            for each in filenames:
                # 如果文件名是以.jpg或者.png结尾则认为是图片,可以自己添加其他格式的照片
                if each.endswith('.jpg') or each.endswith('.png'):
                    i += 1
                    j += 1
                    # 拼接完整的包含路径的文件名
                    scr = os.path.join(dirpath, each)
                    # 拼接新的完整的包含路径的文件名, tail是文件夹的名字
                    if 'NCP' in each:
                        aaa = each.replace('NCP', '2')
                    elif 'CP' in each:
                        aaa = each.replace('CP', '1')
                    elif 'Normal' in each:
                        aaa = each.replace('Normal', '0')
                    dst = os.path.join(dirpath,  aaa)
                    try:
                        # 重命名图片文件
                        # print(scr,dst)
                        os.rename(scr, dst)
                    except:
                        continue
                else:
                    continue
        print('累计重命名{}张图片'.format(j))


if __name__ == '__main__':
    # 设置起始路径path
    path = './data/raw/'
    # 创建实例对象
    pics = BatchRenamePics(path)
    # 调用实例方法
    pics.rename()
