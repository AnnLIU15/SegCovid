import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
# from torchsummary import summary
from tqdm import tqdm

from datasets.segDataSet import COVID19_SegDataSet_test
from datasets.segDataSetNormalize import COVID19_SegDataSetNormalize_test
from models.model import U2NET
from segConfig import getConfig
from utils.Metrics import Mereics_score
from utils.save_result import saveImage


def test(model, test_loader, device, n_classes, save_seg, model_name):
    # 性能指标
    total_acc, total_ap, total_pre, total_recall, total_f1, total_iou, total_dice = 0, 0, 0, 0, 0, 0, 0
    # 固定参数
    model.eval()
    tmp_confusion_matrix = np.zeros(shape=(n_classes, n_classes))
    # 加权和参数
    d = torch.Tensor([3.5, 2.5, 1, 1, 1, 1, 1])
    add_lesion = -4.1
    print('coefficient:', d)
    print('add:', add_lesion)

    with torch.no_grad():
        print('save_dir:', save_seg+'/'+model_name)
        for idx, (imgs, masks, imgs_name) in tqdm(enumerate(test_loader), desc='test', total=len(test_loader)):
            imgs, masks = imgs.to(device), masks.to(device)
            # 获取输出
            d0, d1, d2, d3, d4, d5, d6 = model(imgs)
            # 转为概率
            d0, d1, d2, d3, d4, d5, d6 = nn.Softmax(dim=1)(d0),\
                nn.Softmax(dim=1)(d1), nn.Softmax(dim=1)(d2),\
                nn.Softmax(dim=1)(d3), nn.Softmax(dim=1)(d4),\
                nn.Softmax(dim=1)(d5), nn.Softmax(dim=1)(d6)
            # one-hot独热编码,找最佳权值
            d0_tmp = F.one_hot(d0.clone().argmax(
                dim=1), n_classes).permute(0, 3, 1, 2)
            d1_tmp = F.one_hot(d1.clone().argmax(
                dim=1), n_classes).permute(0, 3, 1, 2)
            d2_tmp = F.one_hot(d2.clone().argmax(
                dim=1), n_classes).permute(0, 3, 1, 2)
            d3_tmp = F.one_hot(d3.clone().argmax(
                dim=1), n_classes).permute(0, 3, 1, 2)
            d4_tmp = F.one_hot(d4.clone().argmax(
                dim=1), n_classes).permute(0, 3, 1, 2)
            d5_tmp = F.one_hot(d5.clone().argmax(
                dim=1), n_classes).permute(0, 3, 1, 2)
            d6_tmp = F.one_hot(d6.clone().argmax(
                dim=1), n_classes).permute(0, 3, 1, 2)
            # for 3-classes
            # d=torch.Tensor([1,0,0,0,0,0,0,0])
            # add_lesion = 0
            # or
            # d=torch.Tensor([3.5,1.5,1,1,1,1,1])
            # add_lesion = -2.1

            # for 2-classes
            # d=torch.Tensor([1,1,1,1,1,0,0,0])
            # add_lesion = -4.1
            
            # 加权和
            d_weighted_sum = d0_tmp*d[0]+d1_tmp*d[1]+d2_tmp*d[2]+d3_tmp*d[3]\
                + d4_tmp*d[4]+d5_tmp*d[5]+d6_tmp*d[6]
            d_weighted_sum[:, 1:n_classes, :, :] = d_weighted_sum[:, 1:n_classes, :, :]+add_lesion
            
            # 获取加权和后取最大概率值并保存
            out_mask = d_weighted_sum.argmax(dim=1)
            saveImage(out_mask, imgs_name, save_seg+'/'+model_name, True)

            # 混淆矩阵
            tmp_confusion_matrix += confusion_matrix(masks.clone().detach().cpu().numpy().ravel(
            ), out_mask.clone().detach().cpu().numpy().ravel(), labels=range(n_classes))
            
            # 指标
            order = Mereics_score(out_mask.clone().detach().cpu(
            ).numpy(), masks.clone().detach().cpu().numpy())
            for idx_1 in range(n_classes-1):
                total_acc += order['accuracy_'+str(idx_1+1)]
                total_pre += order['precision_score_'+str(idx_1+1)]
                total_recall += order['recall_score_'+str(idx_1+1)]
                total_f1 += order['f1_score_'+str(idx_1+1)]
                total_ap += order['AP_'+str(idx_1+1)]
                total_iou += order['iou_'+str(idx_1+1)]
                total_dice += order['dice_coff_' + str(idx_1+1)]
            torch.cuda.empty_cache()

    # 平均每个图片的指标
    mAP = total_ap / len(test_loader)/(n_classes-1)
    avg_acc = total_acc / len(test_loader)/(n_classes-1)
    avg_pre = total_pre / len(test_loader)/(n_classes-1)
    avg_recall = total_recall / len(test_loader)/(n_classes-1)
    avg_f1 = total_f1 / len(test_loader)/(n_classes-1)
    avg_iou = total_iou / len(test_loader)/(n_classes-1)
    avg_dice = total_dice / len(test_loader)/(n_classes-1)
    
    print('confusion_matrix:')
    print(tmp_confusion_matrix)
    confusion_col_sum = tmp_confusion_matrix.sum(axis=0)  # axis=1每一行相加
    confusion_acc = np.zeros(tmp_confusion_matrix.shape)
    # 准确率混淆矩阵
    for i in range(tmp_confusion_matrix.shape[0]):
        confusion_acc[:, i] = np.array(tmp_confusion_matrix[:, i]/confusion_col_sum[i])
    print('confusion_acc_matrix:')
    print(confusion_acc)

    return mAP, avg_acc, avg_pre, avg_recall, avg_f1, avg_iou, avg_dice


def main(args):
    device, num_classes, pth, save_seg, test_data_dir, model_name, normalize = \
        args.device, args.num_classes, args.pth, args.save_seg, args.test_data_dir, args.model_name, args.normalize
    # 掩膜输出位置
    if not os.path.exists(save_seg):
        os.makedirs(save_seg)
    # ng = torch.cuda.device_count()
    # print("Available cuda Devices:{}".format(ng))
    # for i in range(ng):
    #     print('device%d:' % i, end='')
    #     print(torch.cuda.get_device_properties(i))

    if device == 'cuda':
        torch.cuda.set_device(0)
        if not torch.cuda.is_available():
            print('Cuda is not available, use CPU to train.')
            device = 'cpu'
    device = torch.device(device)
    # print('===>device:', device)
    torch.cuda.manual_seed_all(0)

    # Load data

    # print('===>Setup Model')
    model = U2NET(in_channels=1, out_channels=num_classes).to(device)
    checkpoint = torch.load(pth)
    # print('===>Loaded Weight')
    model.load_state_dict(checkpoint['model_weights'])
    if normalize:
        SegDataSet = COVID19_SegDataSetNormalize_test
    else:
        SegDataSet = COVID19_SegDataSet_test
    # print('===>Loading dataset')
    test_data_loader = DataLoader(
        dataset=SegDataSet(test_data_dir, n_classes=3), batch_size=1,
        num_workers=8, shuffle=False, drop_last=False)
    '''
    需要显示模型请把下一句取消注释
    To display the model, please uncomment the next sentence
    '''
    # print('===>Display model')
    # summary(model, (1, 512, 512))
    # print('===>Start Testing')
    test_start_time = time.time()

    # 测试
    mAP, avg_acc, avg_pre, avg_recall, avg_f1, avg_iou, avg_dice = test(model=model, test_loader=test_data_loader, device=device,
                                                                        n_classes=num_classes, save_seg=save_seg, model_name=model_name)
    # 输出测试性能指标结果
    print('Test mAP:%.4f\t\taccuracy:%.4f\t\tprecision:%.4f\t\trecall:%.4f\t\tf1_score:%.4f\t\tiou:%.4f\t\tdice:%.4f'
          % (mAP, avg_acc, avg_pre, avg_recall, avg_f1, avg_iou, avg_dice))
    print('This test total cost %.4fs' % (time.time()-test_start_time))
    # 保存测试性能指标结果
    with open('log/save_log/'+model_name+'testResult.txt', 'w') as f:
        print('model_name:', model_name, file=f)
        print('mAP\t\taccuracy\t\tprecision\t\trecall\t\tf1_score\t\tiou\t\tdice:', file=f)
        print('%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f' %
              (mAP, avg_acc, avg_pre, avg_recall, avg_f1, avg_iou, avg_dice), file=f)


if __name__ == '__main__':
    '''
    测试流程与训练一致,如有不懂请对照segTrain观看
    '''
    args = getConfig('test')
    main(args)
