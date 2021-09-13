

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


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import *


def double_conv(in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),     
    )
def SizeOutputTensor(input_image, kernel, stride, padding, pool, pool_stride):
     output_image = (input_image - kernel + 2 * padding)/stride + 1
     output_image = (output_image - pool) / pool_stride + 1
     print(output_image)
     if output_image > 2:
         SizeOutputTensor(output_image, kernel, stride, padding, pool, pool_stride)
     

class FoInternNet(nn.Module):
    def __init__(self,input_size,n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        
        self.dconv_down1 = double_conv(3, 64)
        SizeOutputTensor(224,3,1,1,2,2)
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
        print(x.shape)
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        print(x.shape)
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        print(x.shape)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        print(x.shape)
        
        x = self.dconv_down4(x)
        x = self.upsample(x)    
        x = torch.cat([x, conv3], dim=1)
        print(x.shape)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)    
        x = torch.cat([x, conv2], dim=1)    
        print(x.shape)
        
        x = self.dconv_up2(x)
        x = self.upsample(x)    
        x = torch.cat([x, conv1], dim=1)   
        print(x.shape)
        
        x = self.dconv_up1(x)
        print(x.shape)
      
        x = self.conv_last(x)
        x = nn.Softmax(dim=1)(x)
        print(x.shape)
        print("-------------------------------")
        return x
    
   """ 
    
    
    
    
    
    
    
    
    
    
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import *


def convDown(in_channels,out_channels, kernel_size, padding):
        return nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),     
            nn.MaxPool2d(2,2)
            
    )
def convUp(in_channels,out_channels, kernel_size, padding):
        return nn.Sequential(
            
            nn.Conv2d(in_channels, out_channels, kernel_size,padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),    
            nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
    )    

class FoInternNet(nn.Module):
    def __init__(self,input_size,n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        
        self.conv1  = convDown(3, 64, 3, 1)
        self.conv2  = convDown(64, 128, 3, 1)
        self.conv3  = convDown(128, 256, 3, 1)
        self.conv4  = convDown(256, 512, 3, 1)  
        
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)

        self.upconv3 = convUp(512, 256, 3, 1)
        self.upconv2 = convUp(128 + 256, 128, 3, 1)
        self.upconv1 = convUp(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, x):
        print(x.shape)
        conv1 = self.conv1(x)
        print(conv1.shape)
        conv2 = self.conv2(conv1)
        print(conv2.shape)
        conv3 = self.conv3(conv2)    
        print(conv3.shape)
        conv4 = self.conv4(conv3)    
        print(conv4.shape)        

        upconv3 = self.upconv3(conv4)
        print(upconv3.shape)
        upconv2 = self.upconv2(upconv3)
        upconv2 = torch.cat([upconv3, conv2], 1)
        print(upconv2.shape)
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        print(upconv1.shape)

        x = self.conv_last(upconv1)
        print(x.shape)
        x = nn.Softmax(dim=1)(x)
        print(x.shape)
        print("----------------------------")
        return x
    """
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import *


def double_conv(in_channels,out_channels,mid_channels=None):
    if not mid_channels:
        mid_channels = out_channels
        return nn.Sequential(

            nn.Conv2d(in_channels, mid_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),        
    )
def yazdir(x):
    d =1
    #print(x)
    
    
class FoInternNet(nn.Module):
    def __init__(self,input_size,n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)  
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)
        self.conv_last = nn.Conv2d(64, n_classes, 1)
         
             
    def forward(self, x):
        yazdir(x.shape)
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        yazdir(x.shape)
        
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        yazdir(x.shape)
        
        conv3 = self.dconv_down3(x)
        yazdir(conv3.shape)
        yazdir("dconv_down3")     
        x = self.maxpool(conv3)   
        yazdir(x.shape)
        
        x = self.dconv_down4(x)
        yazdir(x.shape)
        yazdir("dconv_down4")        
        x = self.upsample(x)    
        yazdir(x.shape)
        yazdir("upsample")
        x = torch.cat([x, conv3], dim=1)
        yazdir(x.shape)
        
        x = self.dconv_up3(x)
        yazdir(x.shape)
        yazdir("dconv_up3")        
        x = self.upsample(x)    
        x = torch.cat([x, conv2], dim=1)    
        yazdir(x.shape)
        
        x = self.dconv_up2(x)
        x = self.upsample(x)    
        x = torch.cat([x, conv1], dim=1)   
        yazdir(x.shape)
        
        x = self.dconv_up1(x)

        yazdir(x.shape)

        
        x = self.conv_last(x)
        x = nn.Softmax(dim=1)(x)
        yazdir(x.shape)
        yazdir("---------------------------------------")
        return x
    

"""