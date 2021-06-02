import os
import time

import numpy as np
import torch
import torch.nn as nn
from numpy.core.fromnumeric import shape
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import torch.nn.functional as F
from datasets.segDataSetNormalize import COVID19_SegDataSetNormalize_test
from datasets.segDataSet import COVID19_SegDataSet_test
from models.u2net import U2NET
from segConfig import getConfig
from utils.new_Metrics import Mereics_score
from utils.torch2img import saveImage


def test(model, test_loader, device, n_classes, save_seg, model_name):
    # total_loss = 0
    total_acc = 0
    total_ap =0
    total_pre=0
    total_recall=0
    total_f1=0
    total_iou=0
    total_dice=0
    model.eval()
    tmp_matrix = np.zeros(shape=(n_classes, n_classes))

    with torch.no_grad():
        print('save_dir:',save_seg+'/'+model_name)
        for idx, (imgs, masks, imgs_name) in tqdm(enumerate(test_loader), desc='test', total=len(test_loader)):
            imgs, masks = imgs.to(device), masks.to(device)
            d0, d1, d2, d3, d4, d5, d6 = model(imgs)
            d0, d1, d2, d3, d4, d5, d6 = nn.Softmax(dim=1)(d0),\
                nn.Softmax(dim=1)(d1), nn.Softmax(dim=1)(d2),\
                nn.Softmax(dim=1)(d3), nn.Softmax(dim=1)(d4),\
                nn.Softmax(dim=1)(d5), nn.Softmax(dim=1)(d6)
            # d0, d1, d2, d3, d4, d5, d6 = d0[:, 1:n_classes, :, :]*1.01,\
            #     d1[:, 1:n_classes, :, :]*1.01, d2[:, 1:n_classes, :, :]*1.01,\
            #     d3[:, 1:n_classes, :, :]*1.01, d4[:, 1:n_classes, :, :]*1.01,\
            #     d5[:, 1:n_classes, :, :]*1.01, d6[:, 1:n_classes, :, :]*1.01
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
            d=torch.Tensor([3.5,2.5,1,1,1,1,1])
            add_lesion = -4.1


            tmp = d0_tmp*d[0]+d1_tmp*d[1]+d2_tmp*d[2]+d3_tmp*d[3]\
                +d4_tmp*d[4]+d5_tmp*d[5]+d6_tmp*d[6]
            tmp[:, 1:n_classes, :, :] = tmp[:, 1:n_classes, :, :]+add_lesion
            out_mask = tmp.argmax(dim=1)
            saveImage(out_mask,imgs_name,save_seg+'/'+model_name,True)
           
            tmp_matrix += confusion_matrix(masks.clone().detach().cpu().numpy().ravel(
            ), out_mask.clone().detach().cpu().numpy().ravel(), labels=range(n_classes))
            order= Mereics_score(out_mask.clone().detach().cpu().numpy(), masks.clone().detach().cpu().numpy())
            for idx_1 in range(n_classes-1):
                total_acc+=order['accuracy_'+str(idx_1+1)]
                total_pre+=order['precision_score_'+str(idx_1+1)]
                total_recall+=order['recall_score_'+str(idx_1+1)]
                total_f1+=order['f1_score_'+str(idx_1+1)]
                total_ap+=order['AP_'+str(idx_1+1)]
                total_iou+=order['iou_'+str(idx_1+1)]
                total_dice+=order['dice_coff_' +str(idx_1+1)]
            # total_loss += loss.clone().detach().cpu().numpy()
            torch.cuda.empty_cache()
    # avg_loss = total_loss / len(test_loader)
    mAP = total_ap / len(test_loader)/(n_classes-1)
    avg_acc = total_acc / len(test_loader)/(n_classes-1)
    avg_pre=total_pre/ len(test_loader)/(n_classes-1)
    avg_recall=total_recall/ len(test_loader)/(n_classes-1)
    avg_f1=total_f1/ len(test_loader)/(n_classes-1)
    avg_iou=total_iou/ len(test_loader)/(n_classes-1)
    avg_dice=total_dice/ len(test_loader)/(n_classes-1)
    print('coefficient:',d)
    print('add:', add_lesion)

    print(tmp_matrix)
    tmp = tmp_matrix.sum(axis=0)  # axis=1每一行相加
    print_data = np.zeros(tmp_matrix.shape)
    for i in range(tmp_matrix.shape[0]):
        print_data[:,i] = np.array(tmp_matrix[:,i]/tmp[i])
    print(print_data)


    return mAP,avg_acc,avg_pre,avg_recall,avg_f1,avg_iou,avg_dice


def main(args):
    device, num_classes, pth, save_seg, test_data_dir, model_name ,normalize= \
        args.device, args.num_classes, args.pth, args.save_seg, args.test_data_dir, args.model_name,args.normalize

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
        SegDataSet=COVID19_SegDataSetNormalize_test
    else:
        SegDataSet=COVID19_SegDataSet_test    
    # print('===>Loading dataset')
    test_data_loader = DataLoader(
        dataset=SegDataSet(test_data_dir, n_classes=3), batch_size=1,
        num_workers=8, shuffle=False, drop_last=False)
    '''
    需要显示模型请把下一句取消注释
    To display the model, please uncomment the next sentence
    '''
    # print('===>Display model')
    #summary(model, (1, 512, 512))
    # print('===>Start Testing')
    test_start_time = time.time()

    # avg_loss,avg_dice,avg_acc,avg_pre,avg_recall,avg_f1=
    mAP,avg_acc,avg_pre,avg_recall,avg_f1,avg_iou,avg_dice=test(model=model, test_loader=test_data_loader, device=device,
         n_classes=num_classes, save_seg=save_seg, model_name=model_name)

    print('Test mAP:%.4f\t\taccuracy:%.4f\t\tprecision:%.4f\t\trecall:%.4f\t\tf1_score:%.4f\t\tiou:%.4f\t\tdice:%.4f'
            % (mAP,avg_acc,avg_pre,avg_recall,avg_f1,avg_iou,avg_dice))
    print('This test total cost %.4fs' % (time.time()-test_start_time))
    with open('log/save_log/'+model_name+'testResult.txt','w') as f:
        print('model_name:',model_name,file=f)
        print('mAP\t\taccuracy\t\tprecision\t\trecall\t\tf1_score\t\tiou\t\tdice:',file=f)
        print('%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f'%
            (mAP,avg_acc,avg_pre,avg_recall,avg_f1,avg_iou,avg_dice),file=f)
if __name__ == '__main__':
    args = getConfig('test')
    main(args)
