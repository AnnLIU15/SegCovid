
# @Author: ZhaoYang
# @Date:   2021-04-21 15:16:01
# @Last Modified by:   ZhaoYang
# @Last Modified time: 2021-05-25 12:40:28
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary



class UNet2d(nn.Module):
    r'''
    in_channel->黑白channel 1 \
    out_channel->大于2的多类如果需要one-shot操作请转换output为SoftMax(这里二类可以用Sigmoid) \
    init_feature->调网络中间channel数
    '''
    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        super(UNet2d, self).__init__()
        self.out_channels=out_channels
        features = init_features
        self.encoder1 = UNet2d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder4 = UNet2d._block(features * 4, features * 8, name="enc4")
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2d._block(features * 4, features * 8, name="bottleneck")

        # self.upconv4 = nn.ConvTranspose2d(
        #     features * 16, features * 8, kernel_size=2, stride=2
        # )
        # self.decoder4 = UNet2d._block((features * 8) * 2 , features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet2d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet2d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        # enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool3(enc3))

        # dec4 = self.upconv4(bottleneck)
        # dec4 = torch.cat((dec4, enc4), dim=1)
        # dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        outputs = self.conv(dec1)
        # 如果要算交叉熵 nn.Softmax(dim=1)(outputs)
        # return nn.Sigmoid()(outputs) #softmax applies on axis=num_classes
        if self.out_channels>1:
            return nn.Softmax(dim=1)(outputs)
        else:
            return outputs
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

def main():
    ng = torch.cuda.device_count()
    print("Available Devices:{}".format(ng))
    infos = [torch.cuda.get_device_properties(i) for i in range(ng)]
    print(infos)
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('使用的设备是(cpu/cuda):',device)
    model=UNet2d(1,3,64).to(device)
    x=torch.rand(1,1, 160, 160)#要是16的倍数
    x=x.to(device)
    y=model.forward(x)

    print(x.shape,y.shape)
    print(torch.max(x),torch.max(y[:,0,:,:]))
    print(torch.min(x),torch.min(y[:,0,:,:]))
    del y
    summary(model, (1,128,128))
if __name__ == '__main__':
    main()