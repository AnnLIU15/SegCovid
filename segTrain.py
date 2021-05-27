import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from datasets.segDataSet import COVID19_SegDataSet
from models.model import U_Net,R2AttU_Net,NestedUNet
from segConfig import getConfig
from utils.loss import dice_loss
from utils.one_hot import one_hot_mask

def train(model, train_loader, optimizer, device,weight):
    epoch_loss = 0
    model.train()
    for idx, (imgs, masks) in tqdm(enumerate(train_loader), desc='Train', total=len(train_loader)):
        imgs, masks = imgs.to(device), masks.to(device)
        # print('='*30)
        # print(torch.unique(masks))
        optimizer.zero_grad()
        output = model(imgs)
        loss = nn.CrossEntropyLoss(weight=torch.Tensor(
            weight).to(device))(output, masks)
        if loss < 0:
            print(idx, loss)
        # print(idx,loss,output.shape,masks.shape)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.clone().detach().cpu().numpy()
        torch.cuda.empty_cache()
    epoch_loss = epoch_loss / len(train_loader)
    return epoch_loss


def val(model, train_loader, device):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for idx, (imgs, masks) in tqdm(enumerate(train_loader), desc='Validation', total=len(train_loader)):
            imgs, masks = imgs.to(device), masks.to(device)

            output = model(imgs)
            loss = nn.CrossEntropyLoss()(output, masks)

            epoch_loss += loss.clone().detach().cpu().numpy()
            torch.cuda.empty_cache()
    epoch_loss = epoch_loss / len(train_loader)
    return epoch_loss


def main(args):
    device, lrate, num_classes, num_epochs, log_name, batch_size,weight,model_name =\
        args.device, args.lrate, args.num_classes, args.num_epochs, args.log_name, args.batch_size,args.weight,args.model_name
    preTrainedSegModel, save_dir, save_every, start_epoch, train_data_dir, val_data_dir = \
        args.preTrainedSegModel, args.save_dir, args.save_every, args.start_epoch, args.train_data_dir, args.val_data_dir
    save_dir=save_dir+'/'+model_name
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
    model=NestedUNet(in_channels=1, out_channels=num_classes).to(device)
    '''
    需要显示模型请把下一句取消注释
    To display the model, please uncomment the next sentence
    '''
    # summary(model,(1, 512, 512))

    print('===>Setting optimizer and scheduler')
    optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-4, last_epoch=-1)

    # logger
    if not os.path.exists('./log/seg/'):
        os.makedirs('./log/seg/')

    if not preTrainedSegModel == None:
        print('===>Loading Pretrained Model')
        checkpoint = torch.load(preTrainedSegModel)
        model.load_state_dict(checkpoint['model_weights'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
    print('===>Making tensorboard log')
    if log_name == None:
        writer = SummaryWriter(
            './log/seg/'+model_name+time.strftime('%m%d-%H%M', time.localtime(time.time())))
    else:
        writer = SummaryWriter('./log/seg/'+log_name)
    # Load data
    print('===>Loading dataset')
    train_data_loader = DataLoader(
        dataset=COVID19_SegDataSet(train_data_dir, n_classes=3), batch_size=batch_size,
        num_workers=8, shuffle=False, drop_last=False)
    val_data_loader = DataLoader(
        dataset=COVID19_SegDataSet(val_data_dir, n_classes=3), batch_size=batch_size,
        num_workers=8, shuffle=False, drop_last=False)
    print('===>Start Training and Validating')
    print("Start training at epoch = {:d}".format(start_epoch))
    best_performance = [0, np.Inf]
    train_start_time = time.time()

    for epoch in range(start_epoch, start_epoch+num_epochs):
        epoch_begin_time = time.time()
        print("\n"+"="*20+"Epoch[{}:{}]".format(epoch, num_epochs)+"="*20 +
              '\nlr={}\tweight_decay={}'.format(optimizer.state_dict()['param_groups'][0]['lr'],
                                                optimizer.state_dict()['param_groups'][0]['weight_decay']))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        train_loss = train(
            model=model, train_loader=train_data_loader, optimizer=optimizer, device=device,weight=weight)
        val_loss = val(
            model=model, train_loader=val_data_loader, device=device,)
        scheduler.step()
        print('Epoch %d Train Loss:%.4f\t\t\tValidation Loss:%.4f' %
              (epoch, train_loss, val_loss))
        if best_performance[1] > train_loss:
            state = {'epoch': epoch, 'model_weights': model.state_dict(
            ), 'optimizer': optimizer.state_dict(),'scheduler':scheduler.state_dict() }
            torch.save(state, os.path.join(
                save_dir, 'best_epoch_model.pth'.format(epoch)))
            best_performance = [epoch, train_loss]

        if epoch % save_every == 0:
            state = {'epoch': epoch, 'model_weights': model.state_dict(
            ), 'optimizer': optimizer.state_dict(),'scheduler':scheduler.state_dict() }
            torch.save(state, os.path.join(
                save_dir, 'epoch_{}_model.pth'.format(epoch)))
        print('Best epoch:%d\t\t\tloss:%.4f' %
              (best_performance[0], best_performance[1]))
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
    writer.close()

    print('This train total cost %.4fs' % (train_end_time-train_start_time))


if __name__ == '__main__':
    args = getConfig('train')
    main(args)
