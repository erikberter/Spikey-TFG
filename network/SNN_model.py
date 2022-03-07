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



class SNN_Model(nn.Module):

    def __init__(self, n_classes, seq_length, p, uses_ts = False, debug = False):
        super(SNN_Model, self).__init__()
        self.seq_length = seq_length
        self.n_classes = n_classes

        self.uses_ts = uses_ts
        self.debug = debug

        self.features1 = SequentialState(
            nn.Conv2d( 3, 16, kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.features2 = SequentialState(
             LIFCell(p),
        )
        self.features = SequentialState(
            #nn.Conv2d( 3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(8),
            #LIFCell(p),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(16),
            LIFCell(p),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(24),
            LIFCell(p),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(32),
            LIFCell(p),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(48),
            LIFCell(p),
            
            nn.Flatten(),
        )

        weigth_scale = 7
        self.features1[0].weight.data *= weigth_scale

        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d):
                self.features[i].weight.data *= weigth_scale
        

        self.classification = SequentialState(
            LILinearCell(768, n_classes, p),
        )

        #self.classification = SequentialState(
        #    nn.Linear(768, 128, bias = False),
        #    LIFCell(p),
        #    nn.Linear(128, 32, bias = False),
        #    LIFCell(p),
        #    nn.Linear(32, self.n_classes, bias = False),
        #    LICell(dt = 0.001),
        #)

        #self.classification[2].weight.data = torch.abs(self.classification[2].weight.data)




    def forward(self, x):
        batch_size = x.shape[1] if self.uses_ts else x.shape[0]
        voltages = torch.empty(
            self.seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
        )
        sc = None
        sf = None
        sf1,sf2 = None, None
        for ts in range(self.seq_length):
            z = z = x[ts, :] if self.uses_ts else x

            if self.debug:
                print(f"OUT_Z {z[0][0]}")
            
                print(f"Z Max {torch.max(z)}")
                print(f"Z Mean {torch.mean(z)}")
            out_f1, sf1 = self.features1(z, sf1)
            
            out_f2, sf2 = self.features2(out_f1, sf2)
            
            out_f, sf = self.features(out_f2, sf)
            
            out_c, sc = self.classification(out_f, sc)
            if self.debug:
                
                print(f"OUT_F1_Pesos {self.features1[0].weight.data[0]}")
                print(f"OUT_F1:Conv {out_f1[0][0]}")
                print(f"OUT_F2:LIF {out_f2[0][0]}")
                print(f"OUT_F {out_f}")
                print(f"OUT_C {out_c}")
                print(f"OUT_C_Pesos\n {self.classification[4].weight.data[0]}")
                input("Jijiji")

            voltages[ts, :, :] = out_c + 0.001 * torch.randn(
                batch_size, self.n_classes, device=z.device
            )

        
        return voltages


class SNN_ModelT(nn.Module):
    def __init__(self, n_classes, use_encoder = False):
        super(SNN_ModelT, self).__init__()
        
        seq_length = 32
        p=LIFParameters(v_th=torch.as_tensor(0.35))
        self.debug = False
        self.use_encoder = use_encoder
        self.encoder = ConstantCurrentLIFEncoder(seq_length)

        self.snn = SNN_Model(n_classes, seq_length, p, uses_ts=use_encoder, debug= self.debug)
        #self.snn = CNN_Pre_Model(n_classes, seq_length)
        self.decoder = decode

    def forward(self, x):
        if self.debug:
            print(f"Input for the model:\n {x[0][0]} \n\n")
        
        if self.use_encoder:
            x = self.encoder(x)
        
        if self.debug:
            print(f"After Encoding:\n {x[0][0][0]}")
            
            print(f"Max in Encoding {torch.max(x)}")
            print(f"Mean in Encoding {torch.mean(x)} \n")
        x = self.snn(x)
        if self.debug:
            print(f"SNN returns:\n {x[0][0]}\n")
            #input("Waiting for confirmation")
        x = self.decoder(x)
        if self.debug:
            print(f"Decoder returns {x[0]}\n")
            input("Pause")
        return x 
