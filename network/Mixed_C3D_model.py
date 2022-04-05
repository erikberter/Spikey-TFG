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

from network.C3D_model import C3D

class Mixed_C3D(nn.Module):
    """
        Usara 3D pero saltandose completamente el mezclar tiempo
    """

    def __init__(self, num_classes, debug = False):
        super(Mixed_C3D, self).__init__()

        self.debug = debug

        self.features = C3D(num_classes, pretrained=True, no_last=True, untrainable=False)
        self.classification = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        out = self.features(x)
        logits = self.classification(out)
        return logits