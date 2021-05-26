import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from datasets.segDataSet import COVID19_SegDataSet_test
from models.model import U_Net
from segConfig import getConfig



def test(model,train_loader,device):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for idx,(imgs,masks) in tqdm(enumerate(train_loader),desc='Test',total=len(train_loader)):
            imgs,masks=imgs.to(device),masks.to(device)

            output = model(imgs)
            loss=nn.CrossEntropyLoss()(output,masks)

            total_loss += loss.clone().detach().cpu().numpy()
            torch.cuda.empty_cache()
    total_loss = total_loss / len(train_loader)
    return total_loss


def main(args):
    device, num_classes, pth,save_seg ,test_data_dir =args.device,args.num_classes, args.pth,args.save_seg,args.test_data_dir
    
    if not os.path.exists(save_seg):
        os.makedirs(save_seg)
    ng = torch.cuda.device_count()
    print("Available cuda Devices:{}".format(ng))
    for i in range(ng):
        print('device%d:'%i,end='')
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
    print('===>Loading dataset')
    test_data_loader = DataLoader(
        dataset=COVID19_SegDataSet_test(test_data_dir,n_classes=3), batch_size=1,
         num_workers=8, shuffle=False, drop_last=False)
    print('===>Setup Model')
    model = U_Net(in_channels=1, out_channels=num_classes).to(device)
    checkpoint = torch.load(pth)
    model.load_state_dict(checkpoint['model_weights'])
    print('loaded')
    exit(1)
    '''
    需要显示模型请把下一句取消注释
    To display the model, please uncomment the next sentence
    '''
    # summary(model,(1, 512, 512))
    
    print('===>Start Testing')
    test_start_time = time.time()
    total_loss=test(model=model, train_loader=test_data_loader,device=device)
    print('Test Loss:%.2f'%total_loss)
    print('This train total cost %.2fs'%(time.time()-test_start_time))

if __name__ == '__main__':
    args = getConfig('test')
    main(args)
