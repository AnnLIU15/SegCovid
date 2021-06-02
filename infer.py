import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from datasets.segDataSet import COVID19_SegDataSet_test
from datasets.segDataSetNormalize import COVID19_SegDataSetNormalize_test
from models.model import U2NET
from segConfig import getConfig


def test(model, test_loader, device, n_classes, save_seg, model_name):

    model.eval()

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


            torch.cuda.empty_cache()
    # avg_loss = total_loss / len(test_loader)





def main(args):
    device, num_classes, pth, infer_data_dir, test_data_dir, model_name ,normalize= \
        args.device, args.num_classes, args.pth, args.infer_data_dir, args.test_data_dir, args.model_name,args.normalize

    if not os.path.exists(infer_data_dir+'/masks/'):
        os.makedirs(infer_data_dir+'/masks/')
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
    test(model=model, test_loader=test_data_loader, device=device,
         n_classes=num_classes, save_seg=save_seg, model_name=model_name)


if __name__ == '__main__':
    args = getConfig('infer')
    main(args)
