import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_aspp import ResNet_ASPP

"""
    GOFM: It aims to strength the representation of grouping features. It contains two key parts: GOM and GFM.
    GOM is in this module.
"""
class GOFM(nn.Module):
    def __init__(self, in_channel, out_channel, nums, conv_size, layer):
        super(GOFM, self).__init__()
        self.width = int(in_channel/nums)
        self.relu = nn.ReLU()
        self.nums = nums
        self.layer = layer
 
        convchannel = []
        if layer == 1:
            for i in range(nums):
                convchannel.append(nn.Conv2d(self.width, out_channel, 1))
        else:
            convchannel.append(nn.Conv2d(self.width+128, out_channel, 3, padding=1))
            for i in range(nums-1):
                convchannel.append(nn.Conv2d(self.width, out_channel, 3, padding=1))

        convs = []
        baseconv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 2*conv_size+1), padding=(0, conv_size)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(2*conv_size+1, 1), padding=(conv_size, 0)),
            nn.Conv2d(out_channel, out_channel, kernel_size=2*conv_size+1, padding=conv_size)
        )
        for i in range(nums):
            convs.append(baseconv)

        self.allconvchannel = nn.ModuleList(convchannel)
        self.allconvs = nn.ModuleList(convs)
        self.conv_cat = GFM(out_channel, layer)

        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, upsal):
        residual = self.conv_res(x)
        if self.layer == 3 or self.layer == 2:
            upsal = F.interpolate(upsal, scale_factor=2)
        if self.layer == 1:
            out = []
            x1 = x
            spl = torch.split(x1, self.width, 1)
            for i in range(self.nums):
                if i == 0:
                    sp = self.allconvchannel[i](spl[i])
                    sp = self.allconvs[i](sp)
                    out.append(sp)
                else:
                    sp = sp + self.allconvchannel[i](spl[i])
                    sp = self.allconvs[i](sp)
                    out.append(sp)
            residual += self.relu(self.conv_cat(out))
        else:
            out = []
            x1 = x
            spl = torch.split(x1, self.width, 1)
            for i in range(self.nums):
                if i == 0:
                    sp = self.allconvchannel[i](torch.cat((spl[i], upsal), 1))
                    sp = self.allconvs[i](sp)
                    out.append(sp)
                else:
                    sp = sp + self.allconvchannel[i](spl[i])
                    sp = self.allconvs[i](sp)
                    out.append(sp)
            residual += self.relu(self.conv_cat(out))

        return self.relu(residual)


"""
    GFM: It aims to integrate grouping features effectively.
"""
class GFM(nn.Module):
    def __init__(self, channel, layer):
        super(GFM, self).__init__()
        self.layer = layer
        self.conv1 = BasicConv2d(channel, channel, 3)
        self.conv2 = BasicConv2d(channel, channel, 3)
        self.conv3 = BasicConv2d(2*channel, channel, 3)
        self.conv4 = BasicConv2d(channel, channel, 3)
        self.conv5 = BasicConv2d(channel, channel, 3)
        self.conv6 = BasicConv2d(2*channel, channel, 3)
        self.conv7 = BasicConv2d(channel, channel, 3)
        self.conv8 = BasicConv2d(channel, channel, 3)
        self.conv9 = BasicConv2d(2*channel, channel, 3)

    def forward(self, x):
        if self.layer == 2 or self.layer == 1 or self.layer == 3:
            x1 = x[0]
            x2 = x[1]
            xt1 = x1 + self.conv1(x1*x2)
            xt2 = x2 + self.conv2(x2*x1)
            out = self.conv3(torch.cat((xt1, xt2), 1))
        else:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            xt1 = x1 + self.conv1(x1*x4)
            xt4 = x4 + self.conv2(x4*x1)
            out1 = self.conv3(torch.cat((xt1, xt4), 1))
            xt2 = x2 + self.conv4(x2*x3)
            xt3 = x3 + self.conv5(x3*x2)
            out2 = self.conv6(torch.cat((xt2, xt3), 1))
            out1 = out1 + self.conv7(out1*out2)
            out2 = out2 + self.conv8(out2*out1)
            out = self.conv9(torch.cat((out1, out2), 1))     
        return out


"""
   BasicConv2d: conv + batchnorm + relu.
"""
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


"""
    FM: It aims to integrate multi-scale saliency features.
"""
class FM(nn.Module):
    def __init__(self, channel, layer):
        super(FM, self).__init__()
        self.layer = layer
        self.relu = nn.ReLU()
        self.conv0 = BasicConv2d(channel, channel, 3)
        self.conv1 = BasicConv2d(channel, channel, 3)
        self.conv2 = BasicConv2d(channel, channel, 3)
        self.conv3 = BasicConv2d(channel, channel, 3)
        self.conv4 = nn.Sequential(BasicConv2d(2*channel, channel, 3), 
                        BasicConv2d(channel, channel, 3))
        self.conv5 = nn.Sequential(BasicConv2d(3*channel, channel, 3),
                        BasicConv2d(channel, channel, 3))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x1, x2, x3):
        xf1 = self.conv0(x1)

        if self.layer == 5:
            x3 = self.conv3(x3)
            z3 = self.relu(xf1 * x3)
            out = torch.cat((xf1, z3), dim=1)
            out = self.conv4(out)
        elif self.layer == 4:
            xf2 = self.conv1(x2)
            z1 = self.relu(xf1 * xf2)

            xz1 = self.conv2(xf1)
            z2 = self.relu(xz1 * x2)
            
            x3 = self.conv3(x3)
            z3 = self.relu(xf1 * x3)
            out = torch.cat((z1, z2, z3), dim=1)
            out = self.conv5(out)
        else:
            xf2 = F.interpolate(self.conv1(x2), scale_factor=2)
            z1 = self.relu(xf1 * xf2)

            xz2 = F.interpolate(x2, scale_factor=2)
            xz1 = self.conv2(xf1)
            z2 = self.relu(xz1 * xz2)
            
            x3 = self.conv3(x3)
            x3 = F.interpolate(x3, scale_factor=2**(4-self.layer))
            z3 = self.relu(xf1 * x3)
            out = torch.cat((z1, z2, z3), dim=1)
            out = self.conv5(out)
        return out


"""
    DSIO: It aims to cross-optimize saliency features and edge features.
"""
class DSIO(nn.Module):
    def __init__(self, channel, layer):
        super(DSIO, self).__init__()
        self.layer = layer
        self.relu = nn.ReLU()
        self.convf1 = nn.Sequential(
            BasicConv2d(channel, channel, 3),
            BasicConv2d(channel, channel, 3),
        )

        self.convf2 = nn.Sequential(
            BasicConv2d(channel, channel, 3),
            BasicConv2d(channel, channel, 3),
        )

        self.conv3 = BasicConv2d(channel, channel, 3)
        self.convf3 = nn.Sequential(
            BasicConv2d(channel, channel, 3),
            BasicConv2d(channel, channel, 3),
        )

        self.convf4 = nn.Sequential(
            BasicConv2d(channel, channel, 3),
            BasicConv2d(channel, channel, 3),
        )

        self.concat = FM(channel, layer)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2, x3, x4):
        if self.layer == 5:
            x_cat = self.concat(x2, x3, x4)
            x1 = F.interpolate(x1, scale_factor=1/4)
            xf1 = x1 + self.convf1(x2*x1)
            xf2 = x_cat + self.convf2((xf1+x_cat)-(xf1*x_cat))
        elif self.layer == 4:
            x_cat = self.concat(x2, x3, x4)
            xf1 = x1 + self.convf3(x2*x1)
            xf2 = x_cat + self.convf4((xf1+x_cat)-(xf1*x_cat))
        else:
            x_cat = self.concat(x2, x3, x4)
            x1 = F.interpolate(x1, scale_factor=2)
            x1 = self.conv3(x1)
            xf1 = x1 + self.convf3(x2*x1)
            xf2 = x_cat + self.convf4((xf1+x_cat)-(xf1*x_cat))
        return xf1, xf2


"""
    Spv: It aims to supervisor the final output.
"""
class Spv(nn.Module):
    def __init__(self, channel):
        super(Spv, self).__init__()
        self.conv1 = nn.Conv2d(channel, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv2(x)
        return x


"""
    Spv2: It aims to supervisor the middle output.
"""
class Spv2(nn.Module):
    def __init__(self, channel, scale_factor):
        super(Spv2, self).__init__()
        self.conv1 = nn.Conv2d(channel, 1, 1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=self.scale_factor)
        return x


"""
    DIPONet_ResNet: This is our proposed model based on ResNet50.
    # loss1 = bce_loss(spvedge1, gt_egs)
    # loss2 = 0.8*bce_loss(globalsal1, gts) + 0.2*CEL()(globalsal1, gts)
    # loss3 = bce_loss(spvedge2, gt_egs)
    # loss4 = 0.8*bce_loss(globalsal2, gts) + 0.2*CEL()(globalsal2, gts)
    total loss = 0.6*loss1 + 0.6*loss2 + loss3 + loss4
    ASPP dalitation rate: 1 6 12 18
"""
class DIPONet_ResNet(nn.Module):
    def __init__(self, pretrained):
        super(DIPONet_ResNet, self).__init__()
        self.resnet = ResNet_ASPP(nInputChannels=3, os=16, backbone_type='resnet50', pretrained=pretrained)
        self.ref5 = GOFM(512*4, 128, 4, 3, 5)
        self.ref4 = GOFM(256*4, 128, 4, 2, 4)
        self.ref3 = GOFM(128*4, 128, 2, 1, 3)
        self.ref2 = GOFM(64*4, 128, 2, 1, 2)
        self.refedge = GOFM(64*4, 128, 2, 1, 1)

        self.spvedge1 = Spv2(128, 4)
        self.spvsal1 = Spv2(128, 16)

        self.ca5 = DSIO(128, 5)
        self.ca4 = DSIO(128, 4) 
        self.ca3 = DSIO(128, 3)
        self.ca2 = DSIO(128, 2)
        
        self.spvedge2 = Spv(128)
        self.spvsal2 = Spv(128)

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = self.resnet(x)
        globalsal1 = self.spvsal1(x6)

        xf5 = self.ref5(x5, x6)
        xf4 = self.ref4(x4, xf5)
        xf3 = self.ref3(x3, xf4)
        xf2 = self.ref2(x2, xf3)
        edge128 = self.refedge(x2, xf2)

        spvedge1 = self.spvedge1(edge128)

        edge5, xf5 = self.ca5(edge128, xf5, xf5, x6)
        edge4, xf4 = self.ca4(edge5, xf4, xf5, x6)
        edge3, xf3 = self.ca3(edge4, xf3, xf4, x6)
        edge2, xf2 = self.ca2(edge3, xf2, xf3, x6)

        spvedge2 = self.spvedge2(edge2)
        globalsal2 = self.spvsal2(xf2)

        return spvedge1, globalsal1, spvedge2, globalsal2

