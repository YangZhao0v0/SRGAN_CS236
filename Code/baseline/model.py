# This file contains the two neural networks in SRGAN:
# Author: Yang Zhao
# Date: 2019/11/15

import numpy as np
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super().__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])
    def forward(self, x):
        return self.features(x)

class ResBlock(nn.Module):
	def __init__(self, in_channel=64, k=3, s=1):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channel, in_channel, k, stride = s, padding = 1),
			nn.BatchNorm2d(in_channel),
			nn.PReLU(),
			nn.Conv2d(in_channel, in_channel, k, stride = s, padding = 1),
			nn.BatchNorm2d(in_channel)
			)

	def forward(self, x):
		return x + self.net(x)

class UpscalingBlock(nn.Module):
	def __init__(self, in_channel=64, out_channel = 256, k=3, s=1):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channel, out_channel, k, stride = s, padding = 1),
			nn.PixelShuffle(2),
			nn.PReLU()
			)

	def forward(self, x):
		return self.net(x)


class Generator(nn.Module):	
    def __init__(self, num_ResBlock = 4, upFactor = 4):
        super().__init__()
        self.num_ResBlock = num_ResBlock
        self.upFactor = upFactor

        self.conv1 = nn.Conv2d(3, 64, 9, 1, 4)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 3, 9, 1, 4)

        # Intermindiate Residual blocks:
        i = 0
        while (i < num_ResBlock):
        	self.add_module('rb' + str(i+1), ResBlock())
        	i += 1

        # Upscaling blocks:
        i = 1
        while (i < upFactor):
        	self.add_module('upScale' + str(i), UpscalingBlock())
        	i *= 2

    def forward(self, x):
    	x = self.conv1(x)
    	x = self.act1(x)
    	inter_x = x.clone()
    	i = 0
    	while (i < self.num_ResBlock):
    		x = self.__getattr__('rb' + str(i+1))(x)
    		i += 1
    	x = self.conv2(x)
    	x = self.bn2(x)
    	x = x + inter_x

    	i = 1
    	while (i < self.upFactor):
        	x = self.__getattr__('upScale' + str(i))(x)
        	i *= 2
    	x = self.conv3(x)
    	return x

class DisBlock(nn.Module):
	def __init__(self, in_channel, out_channel, k, s):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channel, out_channel, k, s, 1),
			nn.BatchNorm2d(out_channel),
			nn.LeakyReLU(0.05, inplace=True)
			)
	def forward(self, x):
		return self.net(x)


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(3, 64, 3, 1, 1),  # [num, 96, 96, 64]
			nn.LeakyReLU(0.05, inplace=True),
			DisBlock(64, 64, 3, 2),	 # [num, 48, 48, 64]
			DisBlock(64, 128, 3, 1),  # [num, 48, 48, 128]
			DisBlock(128, 128, 3, 2), # [num, 24, 24, 128]
			DisBlock(128, 256, 3, 1), # [num, 24, 24, 256]
			DisBlock(256, 256, 3, 2), # [num, 12, 12, 256]
			DisBlock(256, 512, 3, 1), # [num, 12, 12, 512]
			DisBlock(512, 512, 3, 2), # [num, 6, 6, 512]
			nn.Flatten(),
			nn.Linear(6 * 6 * 512, 1024),
			nn.LeakyReLU(0.05, inplace=True),
			nn.Linear(1024,1),
			nn.Sigmoid()
			)

	def forward(self,x):
		return self.net(x)

gen = Generator()
x = torch.randn(10,3,24,24)
y = gen.forward(x)
dis = Discriminator()
z = dis.forward(y)




