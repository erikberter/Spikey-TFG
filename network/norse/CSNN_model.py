from distutils.log import debug
import torch
import torch.nn as nn

from norse.torch.module import SequentialState

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.lif import  LIFCell

from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch import LICell




def decode(x):
    y_hat, _ = torch.max(x, 0)
    #y_hat = x[-1]
    #log_p_y = torch.nn.functional.log_softmax(y_hat, dim=1)
    return y_hat



class CNN_Model_a(nn.Module):
    def __init__(self, n_classes, seq_length, p, uses_ts = False, debug = False):
        super(CNN_Model_a, self).__init__()
        self.seq_length = seq_length
        self.n_classes = n_classes

        self.uses_ts = uses_ts
        self.debug = debug

        self.features = SequentialState(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Sigmoid(),

            nn.Flatten(),
        )



    def forward(self, x):
        return self.features(x)[0]

class SNN_Model_a(nn.Module):

    def __init__(self, n_classes, seq_length, p, uses_ts = False, debug = False):
        super(SNN_Model_a, self).__init__()
        self.seq_length = seq_length
        self.n_classes = n_classes

        self.uses_ts = uses_ts
        self.debug = debug

        self.classification = SequentialState(
            LILinearCell(1024, n_classes, p),
        )

        #self.classification = SequentialState(
        #    nn.Linear(1024, 256, bias = False),
        #    LIFCell(p),
        #    nn.Linear(256, 64, bias = False),
        #    LIFCell(p),
        #    LILinearCell(64, n_classes, p),
        #)

    def forward(self, x):
        batch_size = x.shape[1] if self.uses_ts else x.shape[0]
        
        voltages = torch.empty(
            self.seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
        )

        sc = None
        for ts in range(self.seq_length):
            z = x[ts, :] if self.uses_ts else x

            out_c, sc = self.classification(z, sc)

            voltages[ts, :, :] = out_c

        return voltages


class CSNN_ModelT(nn.Module):
    def __init__(self, n_classes, use_encoder = False):
        super(CSNN_ModelT, self).__init__()
        
        seq_length = 32
        p=LIFParameters(v_th=torch.as_tensor(0.44))
        self.debug = False
        self.use_encoder = use_encoder
        self.encoder = ConstantCurrentLIFEncoder(seq_length)

        self.cnn = CNN_Model_a(n_classes, seq_length, p, uses_ts=use_encoder, debug= self.debug)
        self.snn = SNN_Model_a(n_classes, seq_length, p, uses_ts=use_encoder, debug= self.debug)

        self.decoder = decode

    def forward(self, x):
        
        x = 7 * self.cnn(x)
        if self.debug:
            print(f"CNN {x[0][:20]}")
            print(f"CNN {x.shape}")
        if self.use_encoder:
            x = self.encoder(x)
        if self.debug:
            print(f"Encoder {x[7][0][:20]}")
            print(f"CNN {x.shape}")
        x = self.snn(x)
        if self.debug:
            print(f"SNN {x[:,0,:]}")
            print(f"SNN {x.shape}")
        x = self.decoder(x)
        if self.debug:
            print(f"Decoder {x}")
            print(f"Decoder {x.shape}")
            input("Pause")
        return x 
