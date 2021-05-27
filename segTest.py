import os
import time

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from datasets.segDataSet import COVID19_SegDataSet_test
from models.model import U_Net,NestedUNet
from segConfig import getConfig
from utils.Metrics import Mereics_score,dice_coef
from utils.torch2img import saveImage
def test(model, test_loader, device,save_seg,model_name):
    total_loss = 0
    total_acc = 0
    total_dice =0
    total_pre=0
    total_recall=0
    total_f1=0
    model.eval()
    with torch.no_grad():
        for idx, (imgs, masks,imgs_name) in enumerate(test_loader) :
            imgs, masks = imgs.to(device), masks.to(device)
            output = model(imgs)
            loss = nn.CrossEntropyLoss()(output, masks)
            tmp=output.clone()
            tmp[:,1:3,:,:]=tmp[:,1:3,:,:]*0.00001
            out_mask=tmp.argmax(dim=1)

            saveImage(tmp,imgs_name,save_seg+model_name)
            dice = dice_coef(out_mask.clone().detach().cpu().numpy(), masks.clone().detach().cpu().numpy())
            order=Mereics_score(out_mask.clone().detach().cpu().numpy(), masks.clone().detach().cpu().numpy())
            total_dice+=dice
            total_acc+=order['accuracy']
            total_pre+=order['precision_score']
            total_recall+=order['recall_score']
            total_f1+=order['f1_score']
            total_loss += loss.clone().detach().cpu().numpy()
            torch.cuda.empty_cache()
    avg_loss = total_loss / len(test_loader)
    avg_dice = total_dice / len(test_loader)
    avg_acc = total_acc / len(test_loader)
    avg_pre=total_pre/ len(test_loader)
    avg_recall=total_recall/ len(test_loader)
    avg_f1=total_f1/ len(test_loader)
    return avg_loss,avg_dice,avg_acc,avg_pre,avg_recall,avg_f1


def main(args):
    device, num_classes, pth, save_seg, test_data_dir,model_name = \
        args.device, args.num_classes, args.pth, args.save_seg, args.test_data_dir,args.model_name

    if not os.path.exists(save_seg):
        os.makedirs(save_seg)
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

    # Load data
    
    print('===>Setup Model')
    model = NestedUNet(in_channels=1, out_channels=num_classes).to(device)
    checkpoint = torch.load(pth)
    print('===>Loaded Weight')
    model.load_state_dict(checkpoint['model_weights'])

    print('===>Loading dataset')
    test_data_loader = DataLoader(
        dataset=COVID19_SegDataSet_test(test_data_dir, n_classes=3), batch_size=1,
        num_workers=8, shuffle=False, drop_last=False)
    '''
    需要显示模型请把下一句取消注释
    To display the model, please uncomment the next sentence
    '''
    print('===>Display model')
    #summary(model, (1, 512, 512))
    print('===>Start Testing')
    test_start_time = time.time()


    avg_loss,avg_dice,avg_acc,avg_pre,avg_recall,avg_f1= test(
        model=model, test_loader=test_data_loader, device=device,save_seg=save_seg,model_name=model_name)
    print('Test Loss:%.4f\t\tdice:%.4f\t\taccuracy:%.4f\t\tprecision:%.4f\t\trecall:%.4f\t\tf1_score:%.4f' 
            % (avg_loss,avg_dice,avg_acc,avg_pre,avg_recall,avg_f1))
    print('This train total cost %.4fs' % (time.time()-test_start_time))
    with open('log/save_log/'+model_name+'testResult.txt','w') as f:
        print('model_name:',model_name,file=f)
        print('Loss\t\tdice\t\taccuracy\t\tprecision\t\trecall\t\tf1_score',file=f)
        print('%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f'% 
            (avg_loss,avg_dice,avg_acc,avg_pre,avg_recall,avg_f1),file=f)
if __name__ == '__main__':
    args = getConfig('test')
    main(args)
