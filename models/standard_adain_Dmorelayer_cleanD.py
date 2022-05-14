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

class AdaIN(nn.Module):
    def __init__(self, num_features, style_dim = 64):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1, use_Adain=False, z_dim=64):
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
        
        self.use_Adain = use_Adain
        if(use_Adain):
            # self.adain1 = AdaIN (D)
            # self.adain2 = AdaIN (D)
            self.adain3 = AdaIN (out_channels, z_dim)

    def forward(self, x, z=None):
        bottleneck = self.conv_reduce.forward(x)
        # bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        
        if(self.use_Adain):
            bottleneck = self.adain3(bottleneck, z)
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
        self.extract_hint = extract_hint
        self.msg = msg

        if feat:
            add_channels = 512
        else:
            add_channels = 0

        add_sketch_channel=0
        if(msg):
            add_sketch_channel = 1

        self.toH = nn.Sequential(nn.Conv2d(4, ngf, kernel_size=7, stride=1, padding=3), nn.LeakyReLU(0.2, True))

        self.to0 = nn.Sequential(nn.Conv2d(1, ngf // 2, kernel_size=3, stride=1, padding=1),  # 256, add skeleton
                                 nn.LeakyReLU(0.2, True))
        self.to1 = nn.Sequential(nn.Conv2d(ngf // 2 + add_sketch_channel, ngf, kernel_size=4, stride=2, padding=1),  # 128
                                 nn.LeakyReLU(0.2, True))
        self.to2 = nn.Sequential(nn.Conv2d(ngf  + add_sketch_channel, ngf * 2, kernel_size=4, stride=2, padding=1),  # 64, add z
                                 nn.LeakyReLU(0.2, True))
        self.to3 = nn.Sequential(nn.Conv2d(ngf * 3 + add_sketch_channel, ngf * 4, kernel_size=4, stride=2, padding=1),  # 32, add hint 
                                 nn.LeakyReLU(0.2, True))
        self.to4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),  # 16
                                 nn.LeakyReLU(0.2, True))

  
        depth = 12
        if(extract_hint):
            self.tunnel4_1 = MultiPrmSequential(*[ResNeXtBottleneck(ngf * 8, ngf * 8, cardinality=32, dilate=1, use_Adain=True, z_dim=nz) for _ in range(depth)])
        else:
            self.tunnel4_1 = MultiPrmSequential(*[ResNeXtBottleneck(ngf * 8, ngf * 8, cardinality=32, dilate=1, use_Adain=True, z_dim=nz) for _ in range(depth)])

        self.tunnel4 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                    #  tunnel4,
        )

        self.tunnel4_2 = nn.Sequential(
                                     nn.Conv2d(ngf * 8 , ngf * 4 * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 64

        def to_rgb(in_channels):
            return nn.Conv2d(in_channels, 3, 1, bias=True)

        if self.msg:
            self.rgb_to_features = nn.ModuleList([to_rgb(ngf*4)])

        self.ske_decoder3 = self.skeleton_decoder(ngf*8, ngf*2) # ->32

        depth = 2
        tunnel = [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=1, use_Adain=True, z_dim=nz) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=2, use_Adain=True, z_dim=nz) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=4, use_Adain=True, z_dim=nz) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=2, use_Adain=True, z_dim=nz),
                   ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=1, use_Adain=True, z_dim=nz)]
        self.tunnel3_1 = MultiPrmSequential(*tunnel)

        self.tunnel3 = nn.Sequential(nn.Conv2d(ngf * 8 + ngf*2, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                    #  tunnel3,
        )
        self.tunnel3_2 = nn.Sequential(
                                    nn.Conv2d(ngf * 4, ngf * 2 * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 128

        if self.msg:
            rgb = to_rgb(ngf*2)
            self.rgb_to_features.append(rgb)


        self.ske_decoder2 = self.skeleton_decoder(ngf*4 + ngf*2, ngf) # ->64
        
        depth=1
        tunnel = [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=1, use_Adain=True, z_dim=nz) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=2, use_Adain=True, z_dim=nz) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=4, use_Adain=True, z_dim=nz) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=2, use_Adain=True, z_dim=nz),
                   ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=1, use_Adain=True, z_dim=nz)]
        self.tunnel2_1 = (MultiPrmSequential(*tunnel))

        self.tunnel2 = nn.Sequential(nn.Conv2d(ngf * 4 + ngf, ngf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                    #  tunnel2,
        )
        self.tunnel2_2 = nn.Sequential(
                                     nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        if self.msg:
            
            rgb = to_rgb(ngf)
            self.rgb_to_features.append(rgb)

        self.ske_decoder1 = self.skeleton_decoder(ngf*2 + ngf , ngf//2) # ->128

        tunnel = [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=1, use_Adain=True, z_dim=nz)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=2, use_Adain=True, z_dim=nz)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=4, use_Adain=True, z_dim=nz)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=2, use_Adain=True, z_dim=nz),
                   ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=1, use_Adain=True, z_dim=nz)]
        self.tunnel1_1 = MultiPrmSequential(*tunnel)

        self.tunnel1 = nn.Sequential(nn.Conv2d(ngf * 2 +ngf//2, ngf, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
        )
                                    #  tunnel1,
        self.tunnel1_2 = nn.Sequential(
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
        # print(self.msg)
        if not self.msg:
            x0 = self.to0(sketch)
            x1 = self.to1(x0) 
            x2 = self.to2(x1) 
            x3 = self.to3(torch.cat([x2, hint], 1)) 
        else:
            input = sketch[0]
            x0 = self.to0(input)
            input = torch.cat([x0, sketch[1]], 1)
            x1 = self.to1(input)
            input = torch.cat([x1, sketch[2]], 1)
            x2 = self.to2(input)
            input = torch.cat([x2, hint, sketch[3]], 1)
            x3 = self.to3(input)  # !

        x4 = self.to4(x3)

        x = self.tunnel4(x4)
        if self.extract_hint:
            extract_feat_hint = self.extract_hint(hint)
            x = self.tunnel4_1(x,extract_feat_hint, z)
        else:    
            x = self.tunnel4_1(x, z)
        x = self.tunnel4_2(x)


        if self.msg:
            output = []
            output.append(torch.tanh(self.rgb_to_features[0](x)))

        ske = self.ske_decoder3(x4)
        x = self.tunnel3(torch.cat([x, x3, ske], 1))
        x = self.tunnel3_1(x, z)
        x = self.tunnel3_2(x)
        if self.msg:
            output.append(torch.tanh(self.rgb_to_features[1](x)))

        ske = self.ske_decoder2(torch.cat([ske, x3], 1))
        x = self.tunnel2(torch.cat([x, x2, ske], 1))
        x = self.tunnel2_1(x, z)
        x = self.tunnel2_2(x)
        if self.msg:
            output.append(torch.tanh(self.rgb_to_features[2](x)))

        ske = self.ske_decoder1(torch.cat([ske, x2], 1))
        x = self.tunnel1(torch.cat([x, x1, ske], 1))
        x = self.tunnel1_1(x, z)
        x = self.tunnel1_2(x)

        ske = self.ske_decoder0(torch.cat([ske, x1], 1))
        x = torch.tanh(self.exit(torch.cat([x, x0,ske], 1)))
        if self.msg:
            output.append(x)


        if self.msg:
            if skeleton_output:
                ske = torch.tanh(self.ske_exit(torch.cat([ske, x0], 1)))
                return output, ske
            else:
                return output
        else:
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