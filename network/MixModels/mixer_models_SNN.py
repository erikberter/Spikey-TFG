import torch
import torch.nn as nn

import torchvision.models as models

from network.own.C3NN_Base_model import C3DNN_Small

from network.norse.CSNN_model import SNN_Model_Mix

from norse.torch.module import SequentialState

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.lif import  LIFCell

from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch import LICell

def decode(x):
    y_hat = x[-1]
    #y_hat, _ = torch.max(x, 0)
    return y_hat

class MixClassificationBigSNN(nn.Module):

    def __init__(self, input_size, n_classes):
        super(MixClassificationBigSNN, self).__init__()
        self.seq_length = 24
        self.n_classes = n_classes

        p =  LIFParameters(v_th=torch.as_tensor(0.33))

        self.encoder = ConstantCurrentLIFEncoder(self.seq_length)
        self.decode = decode


        self.lin1 = nn.Linear(2 * input_size, 256, bias=False)
        self.lif1 = LIFCell(p)
        self.drop1 = nn.Dropout(0.2)
        self.lin2 = nn.Linear(256, 256, bias=False)
        self.lif2 = LIFCell(p)
        self.drop2 = nn.Dropout(0.2)
        self.lin3 = nn.Linear(256, 256, bias=False)
        self.lif3 = LIFCell(p)
        self.drop3 = nn.Dropout(0.2)
        self.li = LILinearCell(256, n_classes, p)
        
        
        self.feature_scalar = torch.nn.Parameter(torch.ones(1)) 
        self.encoder_scalar = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = 5 * self.encoder_scalar * self.encoder( 2 * self.feature_scalar * x)
        batch_size = x.shape[1]
        
        voltages = torch.empty(
            self.seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
        )

        slif1, slif2,slif3,sout = None, None, None, None
        for ts in range(self.seq_length):
            z = x[ts, :]

            z = 1.2 * self.lin1(z)
            z, slif1 = self.lif1(z, slif1)
            z = self.drop1(z)
            z = 1.2 * self.lin2(z) 
            z, slif2 = self.lif2(z, slif2)
            z = self.drop2(z)
            z = 1.2 * self.lin3(z)
            z, slif3 = self.lif3(z, slif3)
            z = self.drop3(z)
            out_c, sout = self.li(z, sout)


            voltages[ts, :, :] = out_c

        z = self.decode(voltages)
        return z


class MixClassificationBig(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, input_size, n_classes, debug = False):
        super(MixClassificationBig, self).__init__()

        self.debug = debug

        self.classification = nn.Sequential(
            nn.Linear(2 * input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, n_classes),
        )



    def forward(self, x):
        return self.classification(x)


class MixModelDefaultSNN(nn.Module):
    """
        Mixer model composed of spatial, temporal and fusion models
    """

    def __init__(self, num_classes, debug = False):
        super(MixModelDefaultSNN, self).__init__()

        self.debug = debug

        self.spatial =  models.video.r3d_18(pretrained=True)
        self.temporal =  models.video.r3d_18(pretrained=True)
        self.fusion = MixClassificationBigSNN(400,num_classes)

        for param in self.spatial.parameters():
            param.requires_grad = False

        for param in self.temporal.parameters():
            param.requires_grad = False


    def forward(self, x):

        x_spa = self.spatial(x[0])
        x_temp = self.temporal(x[1])

        x_res = torch.cat((x_spa, x_temp), 1)
        return self.fusion(x_res)