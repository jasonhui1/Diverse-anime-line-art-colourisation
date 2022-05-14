from tkinter import E
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
from torch.nn.utils.parametrizations import spectral_norm
import numpy as np
import math
from torch.autograd import Variable


VGG16_PATH = 'resources/vgg16-397923af.pth'



class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, gamma=2,b=1):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        k_size=int(abs((math.log(channel,2)+b / gamma)))
        k_size += (k_size%2) - 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) //2 )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, channel, gamma=2, b=1, no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = eca_layer(channel=channel, gamma=gamma, b=b)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))
        self.CBAM = CBAM(out_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        # bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        
        bottleneck = self.CBAM(bottleneck) #Attention
        x = self.shortcut.forward(x)

        return x + bottleneck

class MultiPrmSequential(nn.Sequential):
    def __init__(self, *args):
        super(MultiPrmSequential, self).__init__(*args)

    def forward(self, input, feat_hint):
        for module in self._modules.values():
            input = module(input, feat_hint)
        return input


class NetG(nn.Module):
    def __init__(self, ngf=64, nz = 64, feat=False, extract_hint=False, msg=True):
        super(NetG, self).__init__()
        self.feat = feat

        self.toH = nn.Sequential(nn.Conv2d(4, ngf, kernel_size=7, stride=1, padding=3), nn.LeakyReLU(0.2, True))

        self.to0 = nn.Sequential(nn.Conv2d(1, ngf // 2, kernel_size=3, stride=1, padding=1),  # 512
                                 nn.LeakyReLU(0.2, True))
        self.to1 = nn.Sequential(nn.Conv2d(ngf // 2, ngf, kernel_size=4, stride=2, padding=1),  # 256
                                 nn.LeakyReLU(0.2, True))
        self.to2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),  # 128
                                 nn.LeakyReLU(0.2, True))
        self.to3 = nn.Sequential(nn.Conv2d(ngf * 3 + nz, ngf * 4, kernel_size=4, stride=2, padding=1),  # 64
                                 nn.LeakyReLU(0.2, True))
        self.to4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),  # 32
                                 nn.LeakyReLU(0.2, True))
  
        depth = 12
        tunnel4 = nn.Sequential(*[ResNeXtBottleneck(ngf * 8, ngf * 8, cardinality=32, dilate=1) for _ in range(depth)])


        self.tunnel4 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel4,
                                     nn.Conv2d(ngf * 8, ngf * 4 * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 64


        self.ske_decoder3 = self.skeleton_decoder(ngf*8, ngf*2) # ->32

        depth = 2
        tunnel = [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=2),
                   ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=1)]

        tunnel3 = nn.Sequential(*tunnel)

        self.tunnel3 = nn.Sequential(nn.Conv2d(ngf * 8 + ngf*2, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel3,
                                     nn.Conv2d(ngf * 4, ngf * 2 * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 128

        self.ske_decoder2 = self.skeleton_decoder(ngf*4 + ngf*2, ngf) # ->64
        
        depth=1
        tunnel = [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=2),
                   ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=1)]
        tunnel2 = nn.Sequential(*tunnel)

        self.tunnel2 = nn.Sequential(nn.Conv2d(ngf * 4 + ngf, ngf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel2,
                                     nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        self.ske_decoder1 = self.skeleton_decoder(ngf*2 + ngf , ngf//2) # ->128

        tunnel = [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=1)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=2)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=4)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=2),
                   ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=1)]
        tunnel1 = nn.Sequential(*tunnel)

        self.ske_decoder1 = self.skeleton_decoder(ngf*2 + ngf , ngf//2) # ->128
        self.tunnel1 = nn.Sequential(nn.Conv2d(ngf * 2 + ngf//2, ngf, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel1,
                                     nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        self.ske_decoder0 = self.skeleton_decoder(ngf + ngf//2 , ngf//2) # ->256

        self.ske_exit = nn.Conv2d(ngf//2 + ngf//2, 1, kernel_size=3, stride=1, padding=1)
        self.exit = nn.Conv2d(ngf + ngf//2, 3, kernel_size=3, stride=1, padding=1) # 

        if (extract_hint):
            self.extract_hint = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=1, padding=1),  nn.LeakyReLU(0.2, True), 
                                        nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
                                        nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True),)

    def skeleton_decoder(self, in_channels, out_channels, ks=3, stride=1, padding=1, scale=2):
        block = nn.Sequential  (nn.Conv2d(in_channels, out_channels * scale * scale, kernel_size=ks, stride=stride, padding=padding), 
                                nn.PixelShuffle(scale),
                                nn.LeakyReLU(0.2, True),
        )

        return block

    def forward(self, sketch, hint, z, skeleton_output=False, sketch_feat=None):
        hint = self.toH(hint)
        x0 = self.to0(sketch)
        x1 = self.to1(x0) 
        x2 = self.to2(x1) 

        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), hint.size(2), hint.size(3))
        x3 = self.to3(torch.cat([x2, hint, z_img], 1))  

        x4 = self.to4(x3)

        x = self.tunnel4(x4)

        ske = self.ske_decoder3(x4)
        x = self.tunnel3(torch.cat([x, x3, ske], 1))

        ske = self.ske_decoder2(torch.cat([ske, x3], 1))
        x = self.tunnel2(torch.cat([x, x2, ske], 1))

        ske = self.ske_decoder1(torch.cat([ske, x2], 1))
        x = self.tunnel1(torch.cat([x, x1, ske], 1))

        ske = self.ske_decoder0(torch.cat([ske, x1], 1))
        x = torch.tanh(self.exit(torch.cat([x, x0,ske], 1)))


        if skeleton_output:
            ske = torch.tanh(self.ske_exit(torch.cat([ske, x0], 1)))
            return x, ske
        else:
            return x


class NetD(nn.Module):
    def __init__(self, ndf=64, feat=False, hsv=False, hint=False):
        super(NetD, self).__init__()
        self.feat = feat
        self.toH = nn.Sequential(nn.Conv2d(4, ndf, kernel_size=7, stride=1, padding=3), nn.LeakyReLU(0.2, True))

        if feat:
            add_channels = 512
        else:
            add_channels = 0
        
        hsv_channel = 0
        self.hint=hint
        if hsv:
            hsv_channel = 3

        self.feed1 = nn.Sequential(nn.Conv2d(3 + hsv_channel, ndf, kernel_size=7, stride=1, padding=3, bias=False),  # 256
                                  nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # 128
                                  nn.LeakyReLU(0.2, True),

                                  ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1, stride=2),  # 64
                                  nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.LeakyReLU(0.2, True),
        )
        self.feed3 = nn.Sequential(
                                  nn.Conv2d(ndf *2 + ndf, ndf *2, kernel_size=3, stride=1, padding=1, bias=False),  # 128
                                  ResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf * 2 , ndf * 2, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf * 2 , ndf * 2, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1, stride=2),  # 32
                                  nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.LeakyReLU(0.2, True),
  
                                  ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1, stride=2)  # 16
                                  )

        self.feed5 = nn.Sequential(nn.Conv2d(ndf * 4 + add_channels, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),  # 8
                                   nn.LeakyReLU(0.2, True),
                                #    ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                #    ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 16
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 4
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),  # 2
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
                                   nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=0, bias=False),  # 1
                                   nn.LeakyReLU(0.2, True)
                                   )

        self.out = nn.Linear(512, 1)

    def forward(self, color, hint=None, sketch_feat=None, ):

        x = self.feed1(color)
        if self.hint:
            hint = self.toH(hint)
            x = self.feed3(torch.cat([x,hint],1))
        x = self.feed5(x)
        out = self.out(x.view(color.size(0), -1))

        return out


class NetF(nn.Module):
    def __init__(self):
        super(NetF, self).__init__()
        vgg16 = M.vgg16()
        vgg16.load_state_dict(torch.load(VGG16_PATH))
        vgg16.features = nn.Sequential(
            *list(vgg16.features.children())[:9]
        )
        self.model = vgg16.features
        self.register_buffer('mean', torch.FloatTensor([0.485 - 0.5, 0.456 - 0.5, 0.406 - 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
            return self.model((images.mul(0.5) - self.mean) / self.std)


class CSTResNext(nn.Module):
    def __init__(self, channels=256):
        super(CSTResNext, self).__init__()

        D = channels // 2

        self.conv_reduce = nn.Conv2d(channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.conv_conv = nn.Conv2d(D, D, kernel_size=1, stride=1)

        self.conv_expand = nn.Conv2d(D, channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.shortcut = nn.Sequential()
        # if stride != 1:
        #     self.shortcut.add_module('shortcut',
        #                              nn.AvgPool2d(2, stride=2))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        
        return x + bottleneck


class CST(nn.Module):
    def __init__(self):
        super(CST, self).__init__()

        tunnel = nn.Sequential(*[CSTResNext(64) for _ in range(5)])

        self.model = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32,64,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,64,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2, True),
            tunnel,
            nn.Conv2d(64,64,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,32,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32,3,kernel_size=1,stride=1),
        )

    def forward(self, img):
        return torch.tanh(self.model(img))