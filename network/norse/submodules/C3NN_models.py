from distutils.log import debug
from traceback import print_tb
import torch
import torch.nn as nn

from norse.torch.module import SequentialState

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.lif import  LIFCell

from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch import LICell



class Fire3D(nn.Module):
    """
        Versión modificada del Squeeze module para conv3D

        Extraido de: https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py
    """
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int, is_sigmoid = False) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True) if not is_sigmoid else nn.Sigmoid()
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True) if not is_sigmoid else nn.Sigmoid()
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True) if not is_sigmoid else nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )



class C3NN_Feature_Extractor(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, debug = False):
        super(C3NN_Feature_Extractor, self).__init__()

        self.debug = debug

        self.features = SequentialState(
            nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(48),
            nn.MaxPool3d(2),
            nn.Conv3d(48, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Sigmoid(),

            nn.Flatten(),
        )



    def forward(self, x):
        out = self.features(x)
        return out[0]


class C3NN_Fire_Feature_Extractor(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, debug = False):
        super(C3NN_Fire_Feature_Extractor, self).__init__()

        self.debug = debug

        self.features = SequentialState(
            nn.Conv3d(3, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool3d(2, stride = 2,),
            Fire3D(96, 16, 64, 64),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(2, stride = 2, ceil_mode= True),
            Fire3D(128, 16, 64, 64),
            nn.BatchNorm3d(128),
            Fire3D(128, 32, 128, 128),
            nn.MaxPool3d(2, ceil_mode= True),
            nn.BatchNorm3d(256),
            Fire3D(256, 32, 128, 128),
            nn.BatchNorm3d(256),
            Fire3D(256, 48, 192, 192),
            nn.MaxPool3d(2, stride = 2, ceil_mode= True),
            nn.BatchNorm3d(384),
            Fire3D(384, 48, 192, 192),
            nn.MaxPool3d(2, stride = 2, ceil_mode= True),
            Fire3D(384, 64, 256, 256),
            nn.BatchNorm3d(512),
            nn.MaxPool3d(2,  ceil_mode= True),
            Fire3D(512, 64, 256, 256, is_sigmoid=True),
            nn.Flatten(),
        )



    def forward(self, x):
        out = self.features(x)
        return out[0]


class C3NN_Feature_Extractor_Big(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, debug = False):
        super(C3NN_Feature_Extractor_Big, self).__init__()

        self.debug = debug

        self.features = SequentialState(
            nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.MaxPool3d((1,2,2)),
            nn.Sigmoid(),

            nn.Flatten(),
        )



    def forward(self, x):
        out = self.features(x)
        return out[0]


class C3DNN_Feature_Extractor(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, debug = False):
        super(C3DNN_Feature_Extractor, self).__init__()

        self.debug = debug

        self.features = SequentialState(
            nn.BatchNorm3d(3),
            nn.Conv3d(3, 32, kernel_size=(3,5,5), stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(64),

            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 128, kernel_size=(1,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            
            nn.MaxPool3d(2),

            nn.Flatten(),
        )



    def forward(self, x):
        out = self.features(x)
        return out[0]


class C3DNN_Small_t(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self,  debug = False):
        super(C3DNN_Small_t, self).__init__()

        self.debug = debug

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size= 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),

            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(1,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            
            nn.MaxPool3d(2),

            nn.Flatten(),
        )



    def forward(self, x):
        return self.features(x)