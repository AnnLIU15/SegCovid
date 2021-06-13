# 新冠肺炎切割

## 运行环境

| Version  | v3.0    20210605             |
| -------- | ------------------------------- |
| 编程语言 | Python                          |
| Cuda版本 | 10.0                            |
| 库       | [requirements](./requirement.txt) |

## 分割神经网络模型

| 网络名称  | 原始文章                                                     |
| --------- | ------------------------------------------------------------ |
| $U^2$-Net | [$U^2$-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/pdf/2005.09007.pdf) |

| 网络代码                                                     |
| ------------------------------------------------------------ |
| [U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/xuebinqin/U-2-Net) |
| **分割网络排行**                                             |
| [Semantic Segmentation](https://paperswithcode.com/task/semantic-segmentation)<br>[Semantic Segmentation on Cityscapes val](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=unet-a-nested-u-net-architecture-for-medical) |

代码流程图

```flow
st=>start: start
rs=>operation: utils/rename_seg_pic
ps=>operation: utils/preprocessSeg
tr=>operation: segTrain
ts=>operation: segTest
vl=>operation: plot_heatmap and visualizeSeg
rc=>operation: utils/rename_clf_pic
pc=>operation: utils/preprocessClf
ic=>operation: infer or infer_multi_process
rd=>operation: Radiomics/exact_radiomics
e=>end: end

st->rs->ps->tr->ts->vl->rc->pc->ic->rd->e
```







### 1. 数据预处理

**根据[test_split_segmentation.txt](./data/test_split_segmentation.txt)与[train_split_segmentation.txt](./data/train_split_segmentation.txt)将图片划分为训练集和测试集**

从[DATASET.md](./data/DATASET.md)中可以知道数据掩膜的灰度值代表意义如下

```pseudocode
mask中label表示如下(以png文件的灰度值区分)
	分类			    灰度值
	背景(BG)			  0
	肺野(LF)			  1
	肺磨玻璃影(GGO)      2
	肺实质(CO)		     3
```

我们分割的重点在于病灶，即肺磨玻璃影与肺实质，因此将背景和肺野归为一个总体的背景类



in_dir 数据目录，其有子目录imgs与masks

out_dir 输出npy数组目录，后续生成子目录imgs与masks+str(n_classes)

```bash
python ./utils/rename_seg_pic.py
python ./utils/preprocessSeg.py --in_dir data/seg/train/ --n_classes 3 --out_dir data/seg/process/train/
python ./utils/preprocessSeg.py --in_dir data/seg/train/ --n_classes 2 --out_dir data/seg/process/train/
python ./utils/preprocessSeg.py --in_dir data/seg/test/ --n_classes 3 --out_dir data/seg/process/test/
python ./utils/preprocessSeg.py --in_dir data/seg/test/ --n_classes 2 --out_dir data/seg/process/test/
```

**注意！！！如果内存不够不要开z-score！！！分类网络有62000张图，开了z-score需要128G memory<br>but it can improve perfermance**

### 2. 分割训练与测试

3-classes **Train**

```
python segTrain.py --model_name U2Net_n --num_classes 3 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test --weight 1 20 20 --lrate 3e-4 --num_epochs 200

python segTrain_U2Net.py --model_name U2Net_n --num_classes 3 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test --log_name U2Net_n0528-2050 --pth output/saved_models/U2Net_n/epoch_50_model.pth
```

3-classes **Test**

```
 python segTest.py --model_name U2Net_n --num_classes 3 --normalize True --test_data_dir ./data/seg/process/test --pth output_zscore/saved_models/U2Net_n/1_20_20_2/bestSegZcore.pth
```

2-classes **Train**

```
python segTrain.py --model_name U2Net_n_2c --num_classes 2 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test --weight 1 20 --lrate 3e-4 --num_epochs 200

python segTrain.py --model_name U2Net_n_2c --num_classes 2 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test --num_epochs 100 --weight 1 20 --lrate 3e-4 --pth output/saved_models/U2Net_n_2c/epoch_150_model.pth --log_name U2Net_n_2c0529-2000
```

2-classes **Test**

```
 python segTest.py --model_name U2Net_n --num_classes 2 --normalize True --test_data_dir ./data/seg/process/test --pth output_zscore/saved_models/U2Net_n/1_20_20_2/bestSegZcore.pth
```

#### 热力图(test中调用不方便)

```bash
python plot_heatmap.py --model_name U2Net_n --out_dir output/segResult/ --num_classes 2
python plot_heatmap.py --model_name U2Net_n --out_dir output/segResult/ --num_classes 3
```

#### 可视化

```
python visualizeSeg.py
```

#### 导出requirements.txt

```
pipreqs . --encoding=utf8 --force
```

#### 影像组学

```
python Radiomics/exact_radiomics.py
```
#### 性能指标

| Name                            | mAP    | mPA    | IoU    | Mean Dice coff(mDC) | accurcy |
| ------------------------------- | ------ | ------ | ------ | ------------------- | ------- |
| Our(without z-score,3 classes)  | 0.6774 | 0.743  | 0.6346 | 0.7070              | 0.8932  |
| Our(z-score,3 classes)          | 0.6906 | 0.7573 | 0.6514 | 0.7204              | 0.8941  |
| Our(z-score,2 classes)          | _      | 0.8842 | _      | _                   | _       |
| UNet(cell)[1]                   | _      | 0.652  | _      | 0.547               | _       |
| DRUNet(cell)[1]                 | _      | 0.658  | _      | 0.562               | _       |
| FCN(cell)[1]                    | _      | 0.637  | _      | 0.553               | _       |
| SegNet(cell)[1]                 | _      | 0.610  | _      | 0.555               | _       |
| DeepLabv3(cell)[1]              | _      | 0.662  | _      | 0.587               | _       |
| Mask R-CNN(GGO+CO)[7]           | 0.5020 | _      | _      | _                   | _       |
| Mask R-CNN(Lesion)[7]           | 0.6192 | _      | _      | _                   | _       |
| Mask R-CNN*(z-score, Lesion)[7] | 0.6602 | _      | _      | _                   | _       |

##### 3-class confusion matrix

z-score

| 0.998  | 0.200 | 0.077 |
| ------ | ----- | ----- |
| 0.002  | 0.747 | 0.154 |
| 0.0003 | 0.053 | 0.767 |

without z-score

| 0.998  | 0.218 | 0.096 |
| ------ | ----- | ----- |
| 0.002  | 0.710 | 0.128 |
| 0.0002 | 0.072 | 0.776 |

##### 2-class confusion matrix (plot行列错误，但是数据是对的)

| 0.996 | 0.115 |
| ----- | ----- |
| 0.003 | 0.885 |





### 3. 分类

#### 预处理分类图片

```
python ./utils/preprocessClf.py --in_dir /home/e201cv/Desktop/covid_data/clf/train /home/e201cv/Desktop/covid_data/clf/val /home/e201cv/Desktop/covid_data/clf/test --out_dir /home/e201cv/Desktop/covid_data/process_clf/train /home/e201cv/Desktop/covid_data/process_clf/val /home/e201cv/Desktop/covid_data/process_clf/test
```

#### 获取掩模
```
python infer.py --infer_data_dirs /home/e201cv/Desktop/covid_data/process_clf/train /home/e201cv/Desktop/covid_data/process_clf/val /home/e201cv/Desktop/covid_data/process_clf/test --pth output_zscore/saved_models/U2Net_n/1_20_20_2/bestSegZcore.pth --num_classes 3 --device cuda
```

#### 获取影像组学
```
python Radiomics/exact_radiomics.py --imgs_dir data/clf/train/imgs data/clf/val/imgs data/clf/test/imgs --masks_dir data/clf/train/masks data/clf/val/masks data/clf/test/masks --out_dir data/clf/train/radiomics data/clf/val/radiomics data/clf/test/radiomics
```