# CNCB数据集

**24 April 2020 |Version 2.3**

| 数据集下载   | [下载链接](http://ncov-ai.big.ac.cn/download?lang=en)        |
| ------------ | ------------------------------------------------------------ |
| 原始文章出处 | [数据集原始文章](https://www.cell.com/cell/fulltext/S0092-8674(20)30551-1?rss=yes) |
| 文章代码     | [code](http://ncov-ai.big.ac.cn/download/code.zip)           |
| 关键词       | 肺炎、新冠肺炎、分割、分类                                   |



## CNCB 分割数据集

| 文件名            | 下载链接                                                     | md5                              |
| ----------------- | ------------------------------------------------------------ | -------------------------------- |
| ct_lesion_seg.zip | [download](http://ncov-ai.big.ac.cn/download/ct_lesion_seg.zip) | 0c0cab8f929c58a2c8749ac879daa52d |

### ct_lesion_seg.zip 文件结构

* mask：掩膜图片

* image：原始图片
* README.txt：数据集的介绍
* .DS_Store：Mac OS保存文件夹的自定义属性的隐藏文件

### 数据集介绍--简述README.txt

```text
# README.txt
DESCRIPTION
This is the dataset used in KISEG: A Three-Stage Segmentation Framework for Multi-level Acceleration of Chest CT Scans from COVID-19 Patients.

1. Folder structure:
    image: 150 CT scans indexed from 0 to 149. Each CT scan folder contains frame jpg files indexed in a serial order.
    mask: 150 CT scans indexed from 0 to 149. Each mask folder contains annotatation png files for the frame jpg file with the same filename and the same CT scan index.
    
2. Annotation and image file explanation:
    frame jpg files and the corresponding mask png files are all with a resolution of 512x512.
    Each pixel of a mask png file is a uint8 number ranged from 0 to 3 represents the pixel-wised label for the corresponding frame jpg file.
    The meaning of label from 0 to 3 are as follow:
        0: Background (BG)
        1: Lung field (LF)
        2: Ground-glass opacity (GGO)
        3: Consolidation (CO)
```

总结

```text
数据集有750张CT切片(image和mask视为一张)，image与mask的子文件夹以患者为分类，image与mask的数据一一对应
图像切片皆来自于新冠肺炎患者
image图片格式为*.jpg，mask图片格式为*.png

mask中label表示如下(以png文件的灰度值区分)
	分类			灰度值
	背景			0
	肺野			1
	肺磨玻璃影     2
	肺实变		   3
```



## CNCB 分类数据集

[下载地址](http://ncov-ai.big.ac.cn/download?lang=en) <!--注：下载速度较慢--> 

| 数据集类          | 文件名                                     | md5                                 |
| ----------------- | ------------------------------------------ | ----------------------------------- |
| NCP(新型冠状肺炎) | COVID19-1.zip~COVID19-31.zip<br>共31个文件 | [NCP.json](./NCP/filemd5.json)      |
| CP(普通肺炎)      | CP-1.zip~CP-32.zip<br>共32个文件           | [CP.json](./CP/CP.json)             |
| Normal(无病对照)  | Normal-1.zip~Normal-27.zip<br/>共32个文件  | [Normal.json](./Normal/Normal.json) |

| 辅助文件                                   | 简介                                                         |
| ------------------------------------------ | ------------------------------------------------------------ |
| [metadata.csv](./NCP/metadata.csv)         | 一些患者的资料，包括性别、是否有重大基础病、肝与肺功能检测结果、病情发展 |
| [unzip_filenames.csv](unzip_filenames.csv) | zip文件信息，如患者ID，扫描ID与扫描ID对应的slice片数         |
| [lesions_slices.csv](lesions_slices.csv)   | 分类数据集中的病变切片位置<br>如：CP/1434/3937/0034.png      |
