import os
import time

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.segDataSet import COVID19_SegDataSet
from datasets.segDataSetNormalize import COVID19_SegDataSetNormalize
from models.u2net import U2NET, U2NETP
from segConfig import getConfig
from utils.Metrics import enhanced_mixing_loss


def muti_c_dice_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, weight, device, num_classes):
    loss0 = enhanced_mixing_loss(
        d0, labels_v, weight, device, alpha=0.5, n_classes=num_classes)
    loss1 = enhanced_mixing_loss(
        d1, labels_v, weight, device, alpha=0.5, n_classes=num_classes)
    loss2 = enhanced_mixing_loss(
        d2, labels_v, weight, device, alpha=0.5, n_classes=num_classes)
    loss3 = enhanced_mixing_loss(
        d3, labels_v, weight, device, alpha=0.5, n_classes=num_classes)
    loss4 = enhanced_mixing_loss(
        d4, labels_v, weight, device, alpha=0.5, n_classes=num_classes)
    loss5 = enhanced_mixing_loss(
        d5, labels_v, weight, device, alpha=0.5, n_classes=num_classes)
    loss6 = enhanced_mixing_loss(
        d6, labels_v, weight, device, alpha=0.5, n_classes=num_classes)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss0, loss


def train(model, train_loader, optimizer, device, weight, num_classes):
    epoch_loss = 0
    model.train()
    for idx, (imgs, masks) in tqdm(enumerate(train_loader), desc='Train', total=len(train_loader)):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        d0, d1, d2, d3, d4, d5, d6 = model(imgs)
        # print(one_hot_mask_.shape,masks.shape)
        loss2, loss = muti_c_dice_loss_fusion(
            d0, d1, d2, d3, d4, d5, d6, masks, weight, device, num_classes)
        # if loss < 0:
        #     print(idx, loss)
        # print(idx,loss,output.shape,masks.shape)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.clone().detach().cpu().numpy()
        torch.cuda.empty_cache()
    epoch_loss = epoch_loss / len(train_loader)
    return epoch_loss


def val(model, train_loader, device, weight, num_classes):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for idx, (imgs, masks) in tqdm(enumerate(train_loader), desc='Validation', total=len(train_loader)):
            imgs, masks = imgs.to(device), masks.to(device)

            d0, d1, d2, d3, d4, d5, d6 = model(imgs)

            loss2, loss = muti_c_dice_loss_fusion(
                d0, d1, d2, d3, d4, d5, d6, masks, weight, device, num_classes)

            epoch_loss += loss.clone().detach().cpu().numpy()
            torch.cuda.empty_cache()
    epoch_loss = epoch_loss / len(train_loader)
    return epoch_loss


def main(args):
    device, lrate, num_classes, num_epochs, log_name, batch_size, weight, model_name =\
        args.device, args.lrate, args.num_classes, args.num_epochs, args.log_name, args.batch_size, args.weight, args.model_name
    preTrainedSegModel, save_dir, save_every, start_epoch, train_data_dir, val_data_dir = \
        args.preTrainedSegModel, args.save_dir, args.save_every, args.start_epoch, args.train_data_dir, args.val_data_dir
    normalize = args.normalize
    save_dir = save_dir+'/'+model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ng = torch.cuda.device_count()
    print("Available cuda Devices:{}".format(ng))
    for i in range(ng):
        print('device%d:' % i, end='')
        print(torch.cuda.get_device_properties(i))

    if device == 'cuda':
        torch.cuda.set_device(0)
        if not torch.cuda.is_available():
            print('Cuda is not available, use CPU to train.')
            device = 'cpu'
    device = torch.device(device)
    print('===>device:', device)
    torch.cuda.manual_seed_all(0)

    print('===>Setup Model')
    # model = U_Net(in_channels=1, out_channels=num_classes).to(device)
    model = U2NET(in_channels=1, out_channels=num_classes).to(device)
    '''
    需要显示模型请把下一句取消注释
    To display the model, please uncomment the next sentence
    '''
    # summary(model,(1, 512, 512))

    print('===>Setting optimizer and scheduler')
    optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=1e-3)
    ''''''
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, T_max=10, eta_min=1e-5, last_epoch=-1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, eta_min=1e-6, last_epoch=-1, T_mult=2)
    # logger
    if not os.path.exists('./log/seg/'):
        os.makedirs('./log/seg/')

    if not preTrainedSegModel == None:
        print('===>Loading Pretrained Model')
        checkpoint = torch.load(preTrainedSegModel)
        model.load_state_dict(checkpoint['model_weights'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']+1
    print('===>Making tensorboard log')
    if log_name == None:
        writer = SummaryWriter(
            './log/seg/'+model_name+time.strftime('%m%d-%H%M', time.localtime(time.time())))
    else:
        writer = SummaryWriter('./log/seg/'+log_name)
    # Load data
    if normalize:
        SegDataSet = COVID19_SegDataSetNormalize
    else:
        SegDataSet = COVID19_SegDataSet
    print('===>Loading dataset')

    train_data_loader = DataLoader(
        dataset=SegDataSet(train_data_dir, n_classes=num_classes), batch_size=batch_size,
        num_workers=8, shuffle=True, drop_last=False)
    val_data_loader = DataLoader(
        dataset=SegDataSet(val_data_dir, n_classes=num_classes), batch_size=batch_size,
        num_workers=8, shuffle=True, drop_last=False)
    print('train_data_loader:', len(train_data_loader))
    print('val_data_loader:', len(val_data_loader))

    print('===>Start Training and Validating')
    print("Start training at epoch = {:d}".format(start_epoch))
    best_train_performance = [0, np.Inf]
    best_val_performance = [0, np.Inf]
    train_start_time = time.time()

    for epoch in range(start_epoch, start_epoch+num_epochs-1):
        epoch_begin_time = time.time()
        print("\n"+"="*20+"Epoch[{}:{}]".format(epoch, start_epoch+num_epochs)+"="*20 +
              '\nlr={}\tweight_decay={}'.format(optimizer.state_dict()['param_groups'][0]['lr'],
                                                optimizer.state_dict()['param_groups'][0]['weight_decay']))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        train_loss = train(
            model=model, train_loader=train_data_loader, optimizer=optimizer, device=device, weight=weight, num_classes=num_classes)
        val_loss = val(
            model=model, train_loader=val_data_loader, device=device, weight=weight, num_classes=num_classes)
        scheduler.step()
        print('Epoch %d Train Loss:%.4f\t\t\tValidation Loss:%.4f' %
              (epoch, train_loss, val_loss))
        if best_train_performance[1] > train_loss and train_loss>0 and epoch>30:
            state = {'epoch': epoch, 'model_weights': model.state_dict(
            ), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(state, os.path.join(
                save_dir, 'best_train_model.pth'.format(epoch)))
            best_train_performance = [epoch, train_loss]

        if best_val_performance[1] > val_loss and val_loss>0 and epoch>30:
            state = {'epoch': epoch, 'model_weights': model.state_dict(
            ), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(state, os.path.join(
                save_dir, 'best_val_model.pth'.format(epoch)))
            best_val_performance = [epoch, val_loss]

            
        if epoch % save_every == 0:
            state = {'epoch': epoch, 'model_weights': model.state_dict(
            ), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(state, os.path.join(
                save_dir, 'epoch_{}_model.pth'.format(epoch)))
        print('Best train loss epoch:%d\t\t\tloss:%.4f' %
              (best_train_performance[0], best_train_performance[1]))
        print('Best val loss epoch:%d\t\t\tloss:%.4f' %
              (best_val_performance[0], best_val_performance[1]))
        '''
        tensorboard visualize
        ---------------------
        train_loss
        val_loss
        '''
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        epoch_time = time.time()-epoch_begin_time
        print('This epoch cost %.4fs, predicting it will take another %.4fs'
              % (epoch_time, epoch_time*(start_epoch+num_epochs-epoch-1)))
    train_end_time = time.time()
    print('This train total cost %.4fs' % (train_end_time-train_start_time))
    writer.close()


if __name__ == '__main__':
    args = getConfig('train')
    main(args)
