from distutils.log import debug
import torch
import torch.nn as nn

from norse.torch.module import SequentialState

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.lif import  LIFCell, LIFRecurrentCell

from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch import LICell




def decode(x):
    y_hat, _ = torch.max(x, 0)
    #y_hat = x[-1]
    #log_p_y = torch.nn.functional.log_softmax(y_hat, dim=1)
    return y_hat


class CNN_Pre_Model(nn.Module):
    def __init__(self, n_classes, seq_length):
        super(CNN_Pre_Model, self).__init__()
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.features = SequentialState(
            nn.Conv2d( 3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            #LIFCell(p),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            #LIFCell(p),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            #LIFCell(p),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #LIFCell(p),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            #LIFCell(p),
            nn.Flatten(),
            nn.Linear(768, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, self.n_classes),
        )

    def forward(self, x):
        batch_size = x.shape[1]
        voltages = torch.empty(
            self.seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
        )
        for ts in range(self.seq_length):
            z = x[ts, :]
            voltages += self.features(z)[0]

        return voltages

class SNNR_Model(nn.Module):

    def __init__(self, n_classes, seq_length, p, uses_ts = False, debug = False):
        super(SNNR_Model, self).__init__()
        self.seq_length = seq_length
        self.n_classes = n_classes

        self.uses_ts = uses_ts
        self.debug = debug

        self.features =  LIFRecurrentCell(
            3*112*112,
            768,
            p=LIFParameters(alpha=100, 
                            v_th=torch.as_tensor(0.3),
                            tau_syn_inv=torch.tensor(1/1e-2),
                            tau_mem_inv=torch.tensor(1/1e-2),
                           ),
            dt=0.001                     
        )

        weigth_scale = 7

        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d):
                self.features[i].weight.data *= weigth_scale
        

        self.recurrent = LIFRecurrentCell(
            768,
            100,
            p=LIFParameters(alpha=100, 
                            v_th=torch.as_tensor(0.3),
                            tau_syn_inv=torch.tensor(1/1e-2),
                            tau_mem_inv=torch.tensor(1/1e-2),
                           ),
            dt=0.001                     
        )
        self.fc_out = torch.nn.Linear(100, self.n_classes, bias=False)
        self.out = LICell(dt=0.001)

        #self.classification[2].weight.data = torch.abs(self.classification[2].weight.data)




    def forward(self, x):
        batch_size = x.shape[1] if self.uses_ts else x.shape[0]
        voltages = torch.empty(
            self.seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
        )
        sc, sf, so = None, None, None

        for ts in range(self.seq_length):
            z = z = x[ts, :] if self.uses_ts else x

            if self.debug:
                print(f"OUT_Z {z[0][0]}")
            
                print(f"Z Max {torch.max(z)}")
                print(f"Z Mean {torch.mean(z)}")
            out_f, sf = self.features(z, sf)
            
            out_c, sc = self.classification(out_f, sc)
            out_z = self.fc_out(out_c)
            out_o, so = self.out(out_z, so)

            if self.debug:
                print(f"OUT_F {out_f}")
                print(f"OUT_C {out_c}")
                input("Jijiji")

            voltages[ts, :, :] = out_o
            #voltages[ts, :, :] = out_c + 0.001 * torch.randn(
            #    batch_size, self.n_classes, device=z.device
            #)

        
        return voltages


class SNNR_ModelT(nn.Module):
    def __init__(self, n_classes, use_encoder = False):
        super(SNNR_ModelT, self).__init__()
        
        seq_length = 32
        p=LIFParameters(v_th=torch.as_tensor(0.35))
        self.debug = False
        self.use_encoder = use_encoder
        self.encoder = ConstantCurrentLIFEncoder(seq_length)
        #self.cnn = CNN_Pre_Model()
        self.snn = SNNR_Model(n_classes, seq_length, p, uses_ts=use_encoder, debug= self.debug)
        #self.snn = CNN_Pre_Model(n_classes, seq_length)
        self.decoder = decode

    def forward(self, x):
        if self.debug:
            print(f"Input for the model:\n {x[0][0]} \n\n")
        # x = self.cnn(x)
        #if self.debug:
        #    print(f"Conv Tot {x[0][0][:100]}")
        #    print(f"Conv Tot {x[0].shape}")
        #    input("Waiting for confirmation")
        #
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
