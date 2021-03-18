import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv2D,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#-------------------------------------------

class Down2D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder):
        super(Down2D,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_builder(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

#-------------------------------------------

class Up2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder, bilinear=True):
        super(Up2D,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = conv_builder(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = conv_builder(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#-------------------------------------------

class Tail2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tail2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#-------------------------------------------

class UNet(nn.Module):
    def __init__(self, stem, down, up, tail, width, conv_builder, n_channels=1, n_classes=2, bilinear=True, dropout_flag=True, cls_location='middle', revise=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.width = width
        self.dropout_flag = dropout_flag
        self.cls_location = cls_location
        self.revise = revise
        factor = 2 if bilinear else 1

        self.inc = stem(n_channels, width[0])
        self.down1 = down(width[0], width[1], conv_builder)
        self.down2 = down(width[1], width[2], conv_builder)
        self.down3 = down(width[2], width[3], conv_builder)
        self.down4 = down(width[3], width[4] // factor, conv_builder)
        self.up1 = up(width[4], width[3] // factor, conv_builder, bilinear=self.bilinear)
        self.up2 = up(width[3], width[2]// factor, conv_builder, bilinear=self.bilinear)
        self.up3 = up(width[2], width[1] // factor, conv_builder, bilinear=self.bilinear)
        self.up4 = up(width[1], width[0], conv_builder, bilinear=self.bilinear)
        self.dropout = nn.Dropout(p=0.5)
        self.outc = tail(width[0],  self.n_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.clssifier = nn.Sequential(
            nn.Linear(width[4] // factor, 64),
            nn.ReLU(True),
            nn.Linear(64, self.n_classes-1)
        ) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        avg_pool = self.avgpool(x5)

        flatten = torch.flatten(avg_pool,1)
        cls_logits = self.clssifier(flatten)

        if self.dropout_flag:
            x = self.dropout(x)
        seg_logits = self.outc(x)
        
        if self.revise:
            b, c, _, _ = seg_logits.size()
            weight = torch.sigmoid(cls_logits)
            weight = (weight > 0.5).float()
            seg_logits[:,1:,...] = seg_logits[:,1:,...] * weight.view(b,c-1,1,1).expand_as(seg_logits[:,1:,...])

        return [seg_logits,cls_logits]



def unet(**kwargs):
    return UNet(stem=DoubleConv2D,
                down=Down2D,
                up=Up2D,
                tail=Tail2D,
                width=[64,128,256,512,1024],
                conv_builder=DoubleConv2D,
                cls_location='middle',
                **kwargs)


def r_unet(**kwargs):
    return UNet(stem=DoubleConv2D,
                down=Down2D,
                up=Up2D,
                tail=Tail2D,
                width=[64,128,256,512,1024],
                conv_builder=DoubleConv2D,
                cls_location='middle',
                revise=True,
                **kwargs)


# if __name__ == "__main__":
  
#   net = m_unet(n_channels=1, n_classes=2)


#   from torchsummary import summary
#   import os 
#   os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# #   summary(net,input_size=(1,512,512),batch_size=1,device='cuda')
#   summary(net.cuda(),input_size=(1,256,256),batch_size=1,device='cuda')
#   import sys
#   sys.path.append('..')
#   from utils import count_params_and_macs
#   count_params_and_macs(net.cuda(),(1,1,256,256))
