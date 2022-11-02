import sys
from turtle import forward
sys.path.append('.')
from mimetypes import init
import torch
import torch.nn as nn
from base import BaseModel


class ResolveModel(BaseModel):
    def __init__(self) -> None:
        super(ResolveModel, self).__init__()
        channel = 64
        kernel_size = 3
        self.input_resolve_conv = nn.Conv2d(4, 32, kernel_size=kernel_size, padding=1, padding_mode='replicate')
        # self.input_resolve_conv = nn.Conv2d(4, 64, kernel_size=kernel_size, padding=1, padding_mode='replicate')
        self.resolve_convs = nn.Sequential(
            nn.Conv2d(32, channel, kernel_size=kernel_size, padding=1, padding_mode='replicate'),
            # nn.Conv2d(64, 128, kernel_size=kernel_size, padding=1, padding_mode='replicate'),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, channel * 2, kernel_size=kernel_size, padding=1, padding_mode='replicate'),
            # nn.Conv2d(128, 128, kernel_size=kernel_size, padding=1, padding_mode='replicate'),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(channel * 2, channel, kernel_size=kernel_size, padding=1, padding_mode='replicate'),
            # nn.ReLU()
            nn.Conv2d(channel * 2, channel, kernel_size=kernel_size, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, 32, kernel_size=kernel_size, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output_resolve_conv = nn.Sequential(
            nn.Conv2d(channel, 4, kernel_size=kernel_size, padding=1, padding_mode='replicate'),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        input_light = torch.max(inputs, dim=1, keepdim=True)[0]
        input_rgbl = torch.cat((inputs, input_light), 1)
        feature_0 = self.input_resolve_conv(input_rgbl)
        feature_1 = self.resolve_convs(feature_0)
        feature_2 = torch.cat((feature_0, feature_1), 1)
        feature_out = self.output_resolve_conv(feature_2)
        # feature_1 = self.resolve_convs(feature_0)
        # feature_out = self.output_resolve_conv(feature_1)

        R = feature_out[:, 0:3, :, :]
        L = feature_out[:, 3:4, :, :]
        return R, L


class LightModel(BaseModel):
    def __init__(self):
        super(LightModel, self).__init__()
        self.relu = nn.ReLU()
        self.avg_pool = nn.MaxPool2d(2, stride=2)
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.input_layer = nn.Conv2d(1, 32, 3, padding=1, padding_mode='replicate')
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(64, 32, 3, stride=1, padding=1, padding_mode='replicate')
        self.conv5 = nn.Conv2d(64, 32, 3, stride=1, padding=1, padding_mode='replicate')
        self.output_layer = nn.Conv2d(64, 1, 3, stride=1, padding=1, padding_mode='replicate')

    def forward(self, x):
        x1 = self.relu(self.input_layer(x))
        x2 = self.avg_pool(x1)
        x3 = self.relu(self.conv1(x2))
        x4 = self.relu(self.conv2(x3))
        x5 = self.relu(self.conv3(x4))
        x6 = self.relu(self.conv4(torch.cat((x4, x5), dim=1)))
        x7 = self.relu(self.conv5(torch.cat((x3, x6), dim=1)))
        x8 = self.up_sample(x7)
        x9 = self.relu(self.output_layer(torch.cat((x1, x8), dim=1)))
        x_output = torch.pow(x, x9)

        return x_output

class MixModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.ResolveModel = ResolveModel()
        self.LightModel = LightModel()

    def forward(self, x):
        R, L = self.ResolveModel(x)
        low_R = R
        low_L = L
        light_L = self.LightModel(low_L)
        light_L_3 = torch.cat((light_L, light_L, light_L), dim=1)

        return low_R * light_L_3, light_L_3, low_R, low_L

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()

    def forward(self, x):
        return
