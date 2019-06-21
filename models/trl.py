import torch
import torch.nn as nn
import torchvision.models as models


class conv2DGroupNormRelu(nn.Module):
    def __init__(
        self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cgr_unit = nn.Sequential(
            conv_mod, nn.GroupNorm(n_groups, int(n_filters)), nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.cgr_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class ResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, int(planes/4), kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(int(planes/4))
        self.conv2 = nn.Conv2d(int(planes/4), int(planes/4), kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(int(planes/4))
        self.conv3 = nn.Conv2d(int(planes/4), planes, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0),
                                        nn.BatchNorm2d(planes))
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.inplanes != self.planes * self.expansion:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        super(UpSampleBlock, self).__init__()

        self.conv_1 = conv2DBatchNormRelu(in_channels=in_channels,
                                          n_filters=in_channels/2,
                                          k_size=1,
                                          stride=1,
                                          padding=0)

        self.conv_2 = conv2DBatchNormRelu(in_channels=in_channels,
                                          n_filters=in_channels/2,
                                          k_size=3,
                                          stride=1,
                                          dilation=2,
                                          padding=2)

        self.conv_3 = conv2DBatchNormRelu(in_channels=in_channels,
                                          n_filters=in_channels/2,
                                          k_size=5,
                                          stride=1,
                                          dilation=1,
                                          padding=2)

        self.conv_4 = conv2DBatchNormRelu(in_channels=in_channels,
                                          n_filters=in_channels/2,
                                          k_size=7,
                                          stride=1,
                                          dilation=1,
                                          padding=3)

        self.sub_pixel = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        # x = torch.cat([self.conv_1(x), self.conv_1(x), self.conv_1(x), self.conv_1(x)], 1)
        x = self.conv_1(x)
        x = self.sub_pixel(x)
        return x


class TAM(nn.Module):
    def __init__(self, in_channels):
        super(TAM, self).__init__()
        self.BU_conv1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0)
        self.BU_conv2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0)
        self.BU_sigmoid = nn.Sigmoid()

        self.downsample1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rb1 = ResidualBlock(in_channels, in_channels)
        self.rb2 = ResidualBlock(in_channels, in_channels)
        self.rb3 = ResidualBlock(in_channels, in_channels)
        self.rb4 = ResidualBlock(in_channels, in_channels)

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.M_sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, fs, fd):
        # temp->B
        temp = self.BU_sigmoid(self.BU_conv1(torch.cat([fs, fd], 1)))
        temp = self.BU_conv2(torch.cat([temp * fd, temp*(1-temp)], 1))

        temp = self.downsample1(temp)
        temp = self.rb1(temp)
        temp = self.downsample2(temp)
        temp = self.rb2(temp)
        temp = self.upsample1(temp)
        temp = self.rb3(temp)
        temp = self.upsample2(temp)
        temp = self.rb4(temp)

        # temp -> M
        temp = self.M_sigmoid(temp)

        temp = self.conv(torch.cat([(1+temp)*fd, (1+temp)*fs], 1))

        return temp


class ResidualBlockDecode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockDecode, self).__init__()
        self.bottleneck1 = ResidualBlock(in_channels, out_channels)
        self.bottleneck2 = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        return x


class TRL(nn.Module):
    def __init__(self, n_classes):
        super(TRL, self).__init__()
        resnet50 = models.resnet50()
        self.conv1 = nn.Sequential(*list(resnet50.children())[:4])
        self.res_2 = resnet50.layer1
        self.res_3 = resnet50.layer2
        self.res_4 = resnet50.layer3
        self.res_5 = resnet50.layer4

        self.upsample_res_5 = UpSampleBlock(2048)
        self.upsample_res_d1 = UpSampleBlock(1024)
        self.upsample_res_d2 = UpSampleBlock(1024)
        self.upsample_res_d3 = UpSampleBlock(512)
        self.upsample_res_d4 = UpSampleBlock(512)
        self.upsample_res_d5 = UpSampleBlock(256)
        self.upsample_res_d6 = UpSampleBlock(256)

        self.TAM_res_d3 = TAM(512)
        self.TAM_res_d4 = TAM(512)
        self.TAM_res_d5 = TAM(256)
        self.TAM_res_d6 = TAM(256)
        self.TAM_res_d7 = TAM(128)
        self.TAM_res_d8 = TAM(128)

        self.res_d1 = ResidualBlockDecode(2048, 1024)
        self.res_d2 = ResidualBlockDecode(3072, 1024)
        self.res_d3 = ResidualBlockDecode(1024, 512)
        self.res_d4 = ResidualBlockDecode(1024, 512)
        self.res_d5 = ResidualBlockDecode(512, 256)
        self.res_d6 = ResidualBlockDecode(512, 256)
        self.res_d7 = ResidualBlockDecode(128, 128)
        self.res_d8 = ResidualBlockDecode(128, 128)

        self.conv_d1 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1)
        self.conv_d2 = nn.Conv2d(in_channels=1024, out_channels=n_classes, kernel_size=1)
        self.conv_d3 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
        self.conv_d4 = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1)
        self.conv_d5 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.conv_d6 = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1)
        self.conv_d7 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
        self.conv_d8 = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x2 = self.res_2(x)
        x3 = self.res_3(x2)
        x4 = self.res_4(x3)
        x = self.res_5(x4)

        x = self.upsample_res_5(x)
        r1 = self.res_d1(torch.cat([x, x4], 1))
        depth_1 = self.conv_d1(r1)
        r2 = self.res_d2(torch.cat([x, r1, x4], 1))
        segmentation_1 = self.conv_d2(r2)

        x = self.upsample_res_d2(r2)
        r1 = self.res_d3(torch.cat([self.TAM_res_d3(x, self.upsample_res_d1(r1)), x3], 1))
        depth_2 = self.conv_d3(r1)
        r2 = self.res_d4(torch.cat([self.TAM_res_d4(x, r1), x3], 1))
        segmentation_2 = self.conv_d4(r2)

        x = self.upsample_res_d4(r2)
        r1 = self.res_d5(torch.cat([self.TAM_res_d5(x, self.upsample_res_d3(r1)), x2], 1))
        depth_3 = self.conv_d5(r1)
        r2 = self.res_d6(torch.cat([self.TAM_res_d6(x, r1), x2], 1))
        segmentation_3 = self.conv_d6(r2)

        x = self.upsample_res_d6(r2)
        r1 = self.res_d7(self.TAM_res_d7(x, self.upsample_res_d5(r1)))
        depth_4 = self.conv_d7(r1)
        r2 = self.res_d8(self.TAM_res_d8(x, r1))
        segmentation_4 = self.conv_d8(r2)

        return [(depth_1, segmentation_1),
                (depth_2, segmentation_2),
                (depth_3, segmentation_3),
                (depth_4, segmentation_4)]


if __name__ == '__main__':
    import time
    from utils import params_size
    x = torch.rand((1, 3, 512, 1024))
    model = TRL(19)
    params_size(model)
    t1 = time.time()
    x = model(x)
    for each in x:
        for i in each:
            print(i.size())
    print(time.time()-t1)
    # t1 = time.time()
    # resnet50 = models.resnet50()
    # params_size(resnet50)
    # print(time.time()-t1)





