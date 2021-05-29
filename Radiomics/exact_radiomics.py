import argparse
from glob import glob
import numpy as np
from radiomics import featureextractor, getTestCase
from skimage.morphology import dilation, erosion,square
import SimpleITK as sitk


def loadimgs(data_path):
    r'''
    注意这里的imgs需要经过z-score处理后的标准化numpy.ndarray
    masks图片需要先处理
    '''
    img_files=sorted(glob(data_path+'/*.npy'))
    img_names=[var[len(data_path):-4] for var in img_files]
    img_list=[]
    for img_file in img_files:
        img_file=np.load(img_file)
        img_list.append(img_file)
    return np.array(img_list),img_names

def loadTotal(imgs_path,masks_path):
    img_list,img_names=loadimgs(imgs_path)
    mask_list,mask_names=loadimgs(masks_path)
    assert img_names==mask_names,'图像不对应'
    return img_list,mask_list,img_names

def extract_radiomics(imgs_path,masks_path,save_path,max_num_iter=1e4):
    img_list,mask_list,img_names=loadTotal(imgs_path,masks_path)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    for idx,img_name in enumerate(img_names):
        np_array_radiomics=np.zeros(shape=(1037,2))
        imgs=img_list[idx]
        masks=mask_list[idx].astype(np.uint8)
        
        if np.unique(masks).shape[0]==1:
            print(idx)
            continue
        else:
            print(idx,np.unique(masks),masks.sum())
        # 膨胀（dilation)
        masks_dilation = dilation(masks, square(5)).astype(np.uint8)
        # 用边长为5的正方形滤波器进行膨胀滤波
        # 原理：一般对二值图像进行操作。找到像素值为1的点，将它的邻近像素点都设置成这个值。1值表示白，0值表示黑，因此膨胀操作可以扩大白色值范围，压缩黑色值范围。一般用来扩充边缘或填充小的孔洞。
        #腐蚀（erosion)
        masks_erosion=erosion(masks, square(2)).astype(np.uint8)
        # 和膨胀相反的操作，将0值扩充到邻近像素。扩大黑色部分，减小白色部分。可用来提取骨干信息，去掉毛刺，去掉孤立的像素。

        dst_data1 = masks_dilation - masks_erosion
        dst_data2 = masks_erosion.copy()
        sitk_img = sitk.GetImageFromArray(imgs)
        sitk_img.SetSpacing((1, 1, 1))
        sitk_mask1 = sitk.GetImageFromArray(dst_data1)
        sitk_mask1.SetSpacing((1, 1, 1))
        sitk_mask2 = sitk.GetImageFromArray(dst_data2)
        sitk_mask2.SetSpacing((1, 1, 1))
        
        result1 = extractor.execute(sitk_img, sitk_mask1)
        result2 = extractor.execute(sitk_img, sitk_mask2)
        with open('./output/radiomics/'+img_name+'.txt','w+') as f:
            for idx,(key,var) in enumerate(result1.items()):
                if idx>=17:
                    print('key:',key,'\nvar:',var,file=f)
                








def main(args):
    extract_radiomics(args.imgs_dir,args.masks_dir,args.out_dir,)
if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--imgs_dir', type=str,
                         default='data/seg/process/train/imgs/')
    parser_.add_argument('--masks_dir', type=str,
                         default='data/seg/process/train/masks3/')
    parser_.add_argument('--out_dir', type=str,
                         default='data/seg/process/train')
    args = parser_.parse_args()
    main(args)
    