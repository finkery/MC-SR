from torch import nn
import math

class SubNorm(nn.Module):
    def __init__(self,dem_min,dem_range):
        super(SubNorm,self).__init__()
        self.dem_range = dem_range
        self.dem_min = dem_min
    
    def forward(self,x):
        x = (x - self.dem_min) / self.dem_range
        return x
    
class AddNorm(nn.Module):
    def __init__(self,dem_min,dem_range):
        super(AddNorm,self).__init__()
        self.dem_range = dem_range
        self.dem_min = dem_min
    
    def forward(self,x):
        x = x * self.dem_range + self.dem_min
        return x

class ResidualBlock(nn.Module):
    def __init__(self,num_filters = 256):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(num_filters,num_filters,kernel_size=3,padding=1,bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters,num_filters,kernel_size=3,padding=1,bias=True)

    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual

class EDSR(nn.Module):
    def __init__(self,num_filters = 256,num_ResdualBlocks=32,dem_min=0,dem_range=8000,scale=2):
        super(EDSR,self).__init__()
        self.scale = scale
        self.dem_min = dem_min
        self.dem_range = dem_range
        self.conv1 = nn.Conv2d(1,num_filters,kernel_size=3,padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_ResdualBlocks)]
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters,1,kernel_size=3,padding=1,bias=True)
        self.conv3 = nn.Conv2d(num_filters,num_filters * 4,kernel_size=3,padding=1,bias=True)
        self.conv4 = nn.Conv2d(num_filters,num_filters * 9,kernel_size=3,padding=1,bias=True)
        if scale == 3:
            self.PixelShuffle = nn.PixelShuffle(3)
        else:
            self.PixelShuffle = nn.PixelShuffle(2)
        self.subNorm = SubNorm(self.dem_min,self.dem_range)
        self.addNorm = AddNorm(self.dem_min,self.dem_range)
    
    def forward(self,x):
        x = self.subNorm(x)
        x = self.conv1(x)
        res = self.res_blocks(x)
        res += x
        if self.scale == 3:
            x = self.conv4(x)
            x = self.PixelShuffle(x)
            x = self.relu(x)
        else:
            for _ in range(int(math.log(self.scale,2))):
                x = self.conv3(x)
                x = self.PixelShuffle(x)
                x = self.relu(x)
        x = self.conv2(x)
        x = self.addNorm(x)
        return x