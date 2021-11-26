

import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import *


def double_conv(in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  
    )
def SizeOutputTensor(input_image, kernel, stride, padding, pool, pool_stride):
     output_image = (input_image - kernel + 2 * padding)/stride + 1
     print(output_image)
     output_image = (output_image - pool) / pool_stride + 1
     print(output_image)
     if output_image > 2:
         SizeOutputTensor(output_image, kernel, stride, padding, pool, pool_stride)
     

class FoInternNet(nn.Module):
    def __init__(self,input_size,n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        
        self.n_classes = n_classes
        
        self.denemeke = SizeOutputTensor(input_size[0],3,1,1,2,2)
        
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        
        
        self.dconv_down4 = double_conv(256, 512)  
        self.maxpool = nn.MaxPool2d(2,2)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)
        
        self.conv_last = nn.Conv2d(64, n_classes, 1)
         
        
    def forward(self, x):
        print(x.shape) #torch.Size([2, 3, 256, 256])
        
        conv1 = self.dconv_down1(x) #torch.Size([2, 64, 256, 256])
        print(conv1.shape)
        x = self.maxpool(conv1) #torch.Size([2, 64, 128, 128])
        print(x.shape)
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2) #torch.Size([2, 128, 64, 64])
        print(x.shape)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3) #torch.Size([2, 256, 32, 32]) 
        print(x.shape)
        
        x = self.dconv_down4(x)
        x = self.upsample(x)    
        x = torch.cat([x, conv3], dim=1) #torch.Size([2, 768, 64, 64])
        print(x.shape)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)    
        x = torch.cat([x, conv2], dim=1) #torch.Size([2, 384, 128, 128])   
        print(x.shape)
        
        x = self.dconv_up2(x)
        x = self.upsample(x)    
        x = torch.cat([x, conv1], dim=1) #torch.Size([2, 192, 256, 256]) 
        print(x.shape)
        
        x = self.dconv_up1(x) #torch.Size([2, 64, 256, 256])
        print(x.shape)
      
        x = self.conv_last(x)
        x = nn.Softmax(dim=1)(x) #torch.Size([2, 2, 256, 256])
        print(x.shape)
        print("-------------------------------")
        return x
