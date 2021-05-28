# 新冠肺炎UNet切割

## 运行环境

| Version  | v0.1    20210523                |
| -------- | ------------------------------- |
| 编程语言 | Python                          |
| Cuda版本 | 10.0                            |
| 库       | [requirements](./requirement.txt) |



## 分割神经网络模型

| 网络名称            | 位置                          | 原始文章                                                     |
| ------------------- | ----------------------------- | ------------------------------------------------------------ |
| UNet                | [U_Net](models/model.py)      | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) |
| RCNN-UNet           | [R2U_Net](models/model.py)    | [Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation](https://arxiv.org/abs/1802.06955) |
| Attention Unet      | [AttU_Net](models/model.py)   | [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999) |
| RCNN-Attention Unet | [R2AttU_Net](models/model.py) | R2U-Net + Attention U-Net                                    |
| UNet++              | [NestedUNet](models/model.py) | [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165) |
| Unet_dict           | [Unet_dict](models/model.py)  | [Enforcing temporal consistency in Deep Learning segmentation of brain MR images](https://arxiv.org/pdf/1906.07160.pdf) |

| 参考代码                                                     |
| ------------------------------------------------------------ |
| [bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets) |
| **分割网络排行**                                             |
| [Semantic Segmentation](https://paperswithcode.com/task/semantic-segmentation)<br>[Semantic Segmentation on Cityscapes val](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=unet-a-nested-u-net-architecture-for-medical) |



### 分割

```bash
python segTrain.py --model_name R2AttU_Net --num_classes 3
```

python segTrain_U2Net.py --model_name U2Net_n --num_classes 3 --normalize True

### 测试

```bash
python segTest.py --model_name R2AttU_Net --num_classes 3 --pth ./output/saved_models/best_epoch_model.pth
```



### 热力图(test中调用不方便)

```bash
python plot_heatmap_3c.py --model_name UNet++ --out_dir output/segResult/UNet++
```



```
python segTest_U2Net.py --pth output/saved_models/U2Net/best_epoch_model.pth --num_classes 3 --model_name U2Net
```

