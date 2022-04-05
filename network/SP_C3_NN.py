from distutils.log import debug
from platform import release
from traceback import print_tb
from sklearn.metrics import classification_report
import torch
import torch.nn as nn

from norse.torch.module import SequentialState

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.lif import  LIFCell

from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch import LICell


class Spatial_NN(nn.Module):
    """
        Usara 3D pero saltandose completamente el mezclar tiempo
    """

    def __init__(self, debug = False):
        super(Spatial_NN, self).__init__()

        self.debug = debug

        self.features = SequentialState(
            # (3, 16, 112, 112)
            nn.Conv3d(3, 64, kernel_size=(1, 5, 5)), # (3, 16, 110, 110)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), # (3, 16, 24, 24)
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3)), # (3, 16, 22, 22)
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3)), # (3, 16, 20, 20)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),# (3, 16, 10, 10)

            nn.BatchNorm3d(128),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3)), # (3, 16, 8, 8)
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=(1, 3, 3)), # (3, 16, 6, 6)

            nn.MaxPool3d(kernel_size=(1, 2, 2)),# (3, 16, 3, 3)
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3)), # (3, 16, 1, 1)
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3)), # (16, 64, 16, 6, 6)
            
            nn.MaxPool3d(kernel_size=(1, 2, 2)),# (3, 16, 3, 3)
            
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 1, kernel_size=(1, 3, 3)), # (3, 16, 1, 1)
            nn.Sigmoid(),
            
        )



    def forward(self, x):
        out = self.features(x)
        #print(f"LA shape temporal es {out[0].shape}")
        return out[0]

class Temporal_NN(nn.Module):
    """
        Usara 3D pero saltandose completamente el mezclar tiempo
    """

    def __init__(self, debug = False):
        super(Temporal_NN, self).__init__()

        self.debug = debug

        self.features = SequentialState(
            # Initial temporal conv
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
            nn.Dropout(0.2),
            nn.Conv3d(128, 128, kernel_size=(1,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm3d(128),
            
            nn.MaxPool3d(2),

            
            
        )



    def forward(self, x):
        out = self.features(x)
        #print(f"LA shape es {out[0].shape}")
        return out[0]


class SP_NN(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, num_classes, debug = False):
        super(SP_NN, self).__init__()

        self.debug = debug

        self.spatial = Spatial_NN()
        self.temporal = Temporal_NN()

        self.classification = SequentialState(
            nn.Flatten(),
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )



    def forward(self, x):

        batch_size = x.shape[0] 
        spatial_x = self.spatial(x)

        spatial_x = torch.reshape(spatial_x, (batch_size*16,1,1,1))
        x = torch.transpose(x, 1, 2)
        
        x = torch.reshape(x, (batch_size*16, x.shape[2], x.shape[3], x.shape[4]))

        x = spatial_x * x
        
        x = torch.reshape(x, (batch_size, 16, x.shape[1], x.shape[2], x.shape[3]))
        x = torch.transpose(x, 1, 2)
        
        temporal_x = self.temporal(x)

        #print(f"Shape {temporal_x.shape}")
        out = self.classification(temporal_x)[0]
        #print(f"Out {out.shape}")
        return out