# 新冠肺炎切割

## 运行环境

| Version  | v2.0    20210602             |
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
python ./utils/preprocessSeg.py --in_dir data/seg/test/ --n_classes 3 --out_dir data/seg/process/test/
```



### 2. 分割训练与测试

3-classes **Train**

```
python segTrain.py --model_name U2Net_n --num_classes 3 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test --weight 1 20 20 --lrate 3e-4 --num_epochs 200

python segTrain_U2Net.py --model_name U2Net_n --num_classes 3 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test --log_name U2Net_n0528-2050 --preTrainedSegModel output/saved_models/U2Net_n/epoch_50_model.pth
```

3-classes **Test**

```
 python segTest.py --model_name U2Net_n --num_classes 3 --normalize True --test_data_dir ./data/seg/process/test --pth output/saved_models/U2Net_n/1_20_20_2_best_in_150/epoch_150_model.pth
```

2-classes **Train**

```
python segTrain.py --model_name U2Net_n_2c --num_classes 2 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test --weight 1 20 --lrate 3e-4 --num_epochs 200

python segTrain.py --model_name U2Net_n_2c --num_classes 2 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test --num_epochs 100 --weight 1 20 --lrate 3e-4 --preTrainedSegModel output/saved_models/U2Net_n_2c/epoch_150_model.pth --log_name U2Net_n_2c0529-2000
```

2-classes **Test**

```
 python segTest.py --model_name U2Net_n --num_classes 2 --normalize True --test_data_dir ./data/seg/process/test --pth output/saved_models/U2Net_n_2c/epoch_70_model.pth
```

#### 热力图(test中调用不方便)

```bash
python plot_heatmap_2c.py --model_name U2Net_n --out_dir output/segResult/
python plot_heatmap_3c.py --model_name U2Net_n --out_dir output/segResult/
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
### 3. 分类
#### 预处理分类图片

```
python ./utils/preprocessClf.py --in_dir /home/e201cv/Desktop/covid_data/clf/train /home/e201cv/Desktop/covid_data/clf/val /home/e201cv/Desktop/covid_data/clf/test --out_dir /home/e201cv/Desktop/covid_data/process_clf/train /home/e201cv/Desktop/covid_data/process_clf/val /home/e201cv/Desktop/covid_data/process_clf/test
```

#### 获取掩模
```
python infer.py --infer_data_dirs /home/e201cv/Desktop/covid_data/process_clf/train /home/e201cv/Desktop/covid_data/process_clf/val /home/e201cv/Desktop/covid_data/process_clf/test --pth output/saved_models/U2Net_n/1_20_20_2_best_in_150/epoch_150_model.pth --num_classes 3 --device cuda
```

#### 获取影像组学
```
python Radiomics/exact_radiomics.py --imgs_dir data/clf/train/imgs data/clf/val/imgs data/clf/test/imgs --masks_dir data/clf/train/masks data/clf/val/masks data/clf/test/masks --out_dir data/clf/train/radiomics data/clf/val/radiomics data/clf/test/radiomics
```