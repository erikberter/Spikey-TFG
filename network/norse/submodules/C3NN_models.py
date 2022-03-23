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