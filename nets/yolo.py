from collections import OrderedDict

import torch
import torch.nn as nn
import cv2
import numpy as np
from nets.CSPdarknet import darknet53


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))



 
class BCP_layer(nn.Module):
    def __init__(self):
        super(BCP_layer, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)  # b*c*208*208
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)  # b*c*104*104
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # b*c*52*52
        self.deconv3 = nn.ConvTranspose2d(32, 16, 4, stride=2,  padding=1)  # b*c*104*104
        self.deconv2 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)   # b*c*208*208
        self.deconv1 = nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1)    # b*c*416*416
        self.pool1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.pool2 = nn.AvgPool2d(3, stride=2, padding=1)
        self.pool3 = nn.AvgPool2d(3, stride=2, padding=1)

    


    def forward(self, x, T_att):
        x = self.conv1(x)
        T_1 = self.pool1(T_att)
        x = nn.ReLU(inplace=True)(T_1 * x)
        x = self.conv2(x)
        T_2 = self.pool2(T_1)
        x = nn.ReLU(inplace=True)(T_2 * x)
        x = self.conv3(x)
        T_3 = self.pool3(T_2)
        x = nn.ReLU(inplace=True)(T_3 * x)
        x = self.deconv3(x)
        x = nn.ReLU(inplace=True)(T_2 * x)
        x = self.deconv2(x)
        x = nn.ReLU(inplace=True)(T_1 * x)
        x = self.deconv1(x)
        #x = nn.ReLU(inplace=True)(T_1 * x)

        return x 



#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

def obtain_t (x_rgb):
    #A = torch.sum(torch.max(x_rgb,  dim=1, keepdim=True)[0]/(x_rgb.shape[2]*x_rgb.shape[3]),(1,2,3),keepdim=True)

    max_rgb = torch.max(x_rgb, dim=1, keepdim=True)[0]
    num = int((x_rgb.shape[2]*x_rgb.shape[3])/1000)
    min_values, _ = torch.topk(max_rgb, num)
    A = torch.mean(min_values)
    t_p = 1- 0.4 * torch.max( (1-torch.nn.functional.max_pool2d(x_rgb, kernel_size=3, stride=1, padding=1)/(1-A)), dim=1)[0]
    t_p = t_p.unsqueeze(dim=1)
    t_pmin = torch.max ((x_rgb-A)/(1-A), dim=1, keepdim=True)[0]
    return A, t_p, t_pmin
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone   = darknet53(pretrained)

        self.conv1      = make_three_conv([512,1024],1024)
        self.SPP        = SpatialPyramidPooling()
        self.conv2      = make_three_conv([512,1024],2048)

        self.upsample1          = Upsample(512,256)
        self.conv_for_P4        = conv2d(512,256,1)
        self.make_five_conv1    = make_five_conv([256, 512],512)

        self.upsample2          = Upsample(256,128)
        self.conv_for_P3        = conv2d(256,128,1)
        self.make_five_conv2    = make_five_conv([128, 256],256)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head3         = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)],128)

        self.down_sample1       = conv2d(128,256,3,stride=2)
        self.make_five_conv3    = make_five_conv([256, 512],512)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head2         = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)],256)

        self.down_sample2       = conv2d(256,512,3,stride=2)
        self.make_five_conv4    = make_five_conv([512, 1024],1024)

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        self.yolo_head1         = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)],512)
        self.BCP = BCP_layer()

    def forward(self, x):


        x_rgb = x[ : , 0:3, : , : ]
        x_t = x[ : , 3, : , : ]       
        A, t_pt, t_pmin = obtain_t(x_rgb)
        x_t=x_t.unsqueeze(dim=1)
        array = x_t.cpu().numpy()

 
        array = np.transpose(array, (0, 2, 3, 1))
        array = np.repeat(array, 3, axis=3)

        hsv_array = np.zeros_like(array)
        for i in range(x_rgb.shape[0]):
            hsv_array[i] = cv2.cvtColor(array[i], cv2.COLOR_BGR2HSV)
        hsv_array =  torch.from_numpy(np.transpose(hsv_array, (0, 3, 1, 2))).cuda()
        T_att = hsv_array[:,2,:,:]
        T_att = T_att.unsqueeze(dim=1)
        T_att = T_att.pow(2)


        t_p = self.BCP(x_rgb, T_att)
        x_rgb = ((x_rgb-A)/t_p) + A
        x = torch.cat((x_rgb,x_t), dim=1)

         






        #  backbone
        x2, x1, x0 = self.backbone(x)

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048 
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.conv2(P5)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.upsample1(P5)
        # 26,26,512 -> 26,26,256
        P4 = self.conv_for_P4(x1)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P4,P5_upsample],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv1(P4)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.upsample2(P4)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P3(x2)
        # 52,52,128 + 52,52,128 -> 52,52,256
        P3 = torch.cat([P3,P4_upsample],axis=1)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        P3 = self.make_five_conv2(P3)

        # 52,52,128 -> 26,26,256
        P3_downsample = self.down_sample1(P3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P3_downsample,P4],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv3(P4)

        # 26,26,256 -> 13,13,512
        P4_downsample = self.down_sample2(P4)
        # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = torch.cat([P4_downsample,P5],axis=1)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.make_five_conv4(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,52,52)
        #---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,26,26)
        #---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,13,13)
        #---------------------------------------------------#
        out0 = self.yolo_head1(P5)

        return out0, out1, out2, t_pt, t_p, t_pmin

