import torch.nn as nn


class res_block(nn.Module):
    def __init__(self, in_size):
        super(res_block, self).__init__()

        self.padding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_size, in_size, 3)
        self.instanceNorm = nn.InstanceNorm2d(in_size)
        # batchNorm = nn.BatchNorm2d(in_size)   # activate if using batchNorm
        self.relu = nn.ReLU(True)

    def convblock(self, x):
        x = self.relu(self.instanceNorm(self.conv(self.padding(x))))
        x = self.instanceNorm(self.conv(self.padding(x)))
        return x

    def forward(self, x):
        return x + self.convblock(x)  # skip connection


class g_net(nn.Module):  # generator net design
    def __init__(self, in_size, out_size, res_num=9):  # default using resnet_9blocks
        super(g_net, self).__init__()

        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_size, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(True),
                  nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.ReLU(True),
                  nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.InstanceNorm2d(256), nn.ReLU(True)]
        for i in range(res_num): layers += [res_block(256)]
        layers += [nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                   nn.InstanceNorm2d(128), nn.ReLU(True),
                   nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                   nn.InstanceNorm2d(64), nn.ReLU(True),
                   nn.ReflectionPad2d(3), nn.Conv2d(64, out_size, 7), nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class d_net(nn.Module):  # discriminator net design
    def __init__(self, input_channels):
        super(d_net, self).__init__()
        in_channels = input_channels
        cfg = [(2, 2), 64, 'LR', 128, 'IN', 'LR', 256, 'IN', 'LR', (1, 1), 512, 'IN', "LR"]
        layers = []
        stride_in_use = 0
        for out_channels in cfg:
            if out_channels == (2, 2):
                stride_in_use = 2
            elif out_channels == (1, 1):
                stride_in_use = 1
            elif out_channels == 'LR':
                layers += [nn.LeakyReLU(0.2, inplace=True)]
            elif out_channels == 'IN':
                layers += [nn.InstanceNorm2d(in_channels)]
            else:
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride_in_use, padding=1)
                layers += [conv2d]
                in_channels = out_channels

        layers += [nn.Conv2d(512, 1, 4, stride=stride_in_use, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)



