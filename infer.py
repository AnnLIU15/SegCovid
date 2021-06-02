import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from datasets.inferDataSet import infer_DataSet
from models.model import U2NET
from segConfig import getConfig


def infer(model, test_loader, device, n_classes, save_seg):

    model.eval()

    with torch.no_grad():
        for idx, (imgs, imgs_name) in tqdm(enumerate(test_loader), desc='infer', total=len(test_loader)):
            imgs = imgs.to(device)
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
            d = torch.Tensor([3.5, 2.5, 1, 1, 1, 1, 1])
            add_lesion = -4.1
            tmp = d0_tmp*d[0]+d1_tmp*d[1]+d2_tmp*d[2]+d3_tmp*d[3]\
                + d4_tmp*d[4]+d5_tmp*d[5]+d6_tmp*d[6]
            tmp[:, 1:n_classes, :, :] = tmp[:, 1:n_classes, :, :]+add_lesion
            out_mask = tmp.argmax(dim=1)
            np.save(save_seg+'/'+imgs_name,
                    out_mask.clone().detach().cpu().numpy().squeeze())

            torch.cuda.empty_cache()
    # avg_loss = total_loss / len(test_loader)


def main(args):
    device, num_classes, pth, infer_data_dirs, model_name = \
        args.device, args.num_classes, args.pth, args.infer_data_dirs, args.model_name

    if device == 'cuda':
        torch.cuda.set_device(0)
        if not torch.cuda.is_available():
            print('Cuda is not available, use CPU to train.')
            device = 'cpu'
    device = torch.device(device)
    print('===>device:', device)
    torch.cuda.manual_seed_all(0)

    # Load data

    print('===>Setup Model')
    model = U2NET(in_channels=1, out_channels=num_classes).to(device)
    checkpoint = torch.load(pth)
    print('===>Loaded Weight')
    model.load_state_dict(checkpoint['model_weights'])
    SegDataSet = infer_DataSet
    print('===>check infer_data_dirs')
    if isinstance(infer_data_dirs, str):
        infer_data_dirs = [infer_data_dirs]
    total_infer_begin = time.time()
    for infer_data_dir in infer_data_dirs:
        imgs_dir = infer_data_dir+'/imgs/'
        masks_save_dir = infer_data_dir+'/masks/'
        if not os.path.exists(masks_save_dir):
            os.makedirs(masks_save_dir)

        print('===>Loading dataset')
        test_data_loader = DataLoader(
            dataset=SegDataSet(imgs_dir), batch_size=1,
            num_workers=8, shuffle=False, drop_last=False)

        print('===>Start infer '+imgs_dir)
        print('===>Save to '+masks_save_dir)
        infer_start_time = time.time()
        infer(model=model, test_loader=test_data_loader, device=device,
              n_classes=num_classes, save_seg=masks_save_dir, model_name=model_name)
        infer_end_time = time.time()
        print('Infer cost %.2fs' % (infer_end_time-infer_start_time))
    total_infer_end = time.time()
    print('Total Infer cost %.2fs' % (total_infer_end-total_infer_begin))


if __name__ == '__main__':
    args = getConfig('infer')
    main(args)
