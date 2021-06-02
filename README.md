# 新冠肺炎UNet切割

## 运行环境

| Version  | v1.0    20210529                |
| -------- | ------------------------------- |
| 编程语言 | Python                          |
| Cuda版本 | 10.0                            |
| 库       | [requirements](./requirement.txt) |

即将加入dropout层

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

in_dir 数据目录，其有子目录imgs与masks

out_dir 输出npy数组目录，后续生成子目录imgs与masks+str(n_classes)

```bash
python datasets/preprocessSeg.py --in_dir data/seg/train/ --n_classes 3 --out_dir data/seg/process/train/
python datasets/preprocessSeg.py --in_dir data/seg/test/ --n_classes 3 --out_dir data/seg/process/test/
```



### 分割

旧版dataloader

```bash
python segTrain.py --model_name R2AttU_Net --num_classes 3
```

新版dataloader3类

```
python segTrain_U2Net.py --model_name U2Net_n --num_classes 3 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test

python segTrain_U2Net.py --model_name U2Net_n --num_classes 3 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test --log_name U2Net_n0528-2050 --preTrainedSegModel output/saved_models/U2Net_n/epoch_50_model.pth

python segTest_U2Net.py --model_name U2Net_n --num_classes 3 --normalize True --test_data_dir ./data/seg/process/test --pth output/saved_models/U2Net_n/epoch_150_model.pth
(coefficient: tensor([3.5000, 2.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
add: -4.1)
```


尝试

python segTrain_U2Net.py --model_name U2Net_n --num_classes 3 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test --weight 1 40 40 --lrate 5e-4 --num_epochs 200



2类

```
python segTrain_U2Net.py --model_name U2Net_n_2c --num_classes 2 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test --num_epochs 160 --weight 1 40


python segTrain_U2Net.py --model_name U2Net_n_2c --num_classes 2 --normalize True --batch_size 4 --train_data_dir ./data/seg/process/train --val_data_dir ./data/seg/process/test --num_epochs 100 --weight 1 40 --lrate 1e-4 --preTrainedSegModel output/saved_models/U2Net_n_2c/epoch_70_model.pth --log_name U2Net_n_2c0529-2000

python segTest_U2Net.py --model_name U2Net_n_2c --num_classes 2 --normalize True --test_data_dir ./data/seg/process/test --pth output/saved_models/U2Net_n_2c/epoch_70_model.pth
```




### 测试

```bash
python segTest.py --model_name R2AttU_Net --num_classes 3 --pth ./output/saved_models/best_epoch_model.pth
```



### 热力图(test中调用不方便)

```bash
python plot_heatmap_3c.py --model_name UNet++ --out_dir output/segResult/UNet++

python plot_heatmap_2c.py --model_name U2Net_n --out_dir output/segResult/
```



```
python segTest_U2Net.py --pth output/saved_models/U2Net/best_epoch_model.pth --num_classes 3 --model_name U2Net
```

