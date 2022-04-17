from distutils.log import debug
from turtle import forward
import torch
import torch.nn as nn

from norse.torch.module import SequentialState

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.lif import  LIFCell

from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch import LICell

from network.norse.submodules.C3NN_models import C3DNN_Small_t
from network.norse.submodules.C3NN_models import C3NN_Feature_Extractor, C3NN_Feature_Extractor_Big, C3NN_Fire_Feature_Extractor, C3DNN_Feature_Extractor

from network.C3D_model import C3D

def decode(x):
    y_hat = x[-1]
    #y_hat, _ = torch.max(x, 0)
    return y_hat



class SNN_Model_a(nn.Module):

    def __init__(self, n_classes, seq_length, p, uses_ts = False, debug = False):
        super(SNN_Model_a, self).__init__()
        self.seq_length = seq_length
        self.n_classes = n_classes

        self.uses_ts = uses_ts
        self.debug = debug

        self.classification = SequentialState(
            nn.Linear(1152, 512),
            LIFCell(p),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            LIFCell(p),
            nn.Dropout(0.2),
            LILinearCell(256, n_classes, p),
        )

    def forward(self, x):
        #return self.classification(x)
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


class SNN_Model_b(nn.Module):

    def __init__(self, n_classes, seq_length, p, uses_ts = False, debug = False):
        super(SNN_Model_b, self).__init__()
        self.seq_length = seq_length
        self.n_classes = n_classes

        self.uses_ts = uses_ts
        self.debug = debug

        self.classification = SequentialState(
            nn.Linear(1024, 512),
            LIFCell(p),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            LIFCell(p),
            nn.Dropout(0.2),
            LILinearCell(256, n_classes, p),
        )

    def forward(self, x):
        #return self.classification(x)
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

        


class C3DSNN_Base(nn.Module):
    def __init__(self, snn, cnn, use_encoder = True, cnn_scaler = 7, encoder_scaler = 1):
        super(C3DSNN_Base, self).__init__()
        
        seq_length = 48
        p = LIFParameters(v_th=torch.as_tensor(0.5))
        self.debug = False
        self.use_encoder = use_encoder
        self.encoder = ConstantCurrentLIFEncoder(seq_length)

        ## Network Parameters

        # Scales the CNN output by a factor to improve the ConstantCurrentLIFEncoder Input
        self.cnn_scaler = cnn_scaler
        self.encoder_scaler = encoder_scaler

        self.cnn = cnn
        self.snn = snn

        self.decoder = decode

    def forward(self, x):
        
        x = self.cnn_scaler * self.cnn(x)
        if self.debug:
            #print(f"CNN {x[0][:20]}")
            print(f"CNN {x.shape}")
        if self.use_encoder:
            x = self.encoder_scaler * self.encoder(x)
        if self.debug:
            #print(f"Encoder {x[7][0][:20]}")
            print(f"Encoder {x.shape}")
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


class C3SNN_ModelT(nn.Module):
    def __init__(self, n_classes, use_encoder = True):
        super(C3SNN_ModelT, self).__init__()
        
        seq_length = 32
        p = LIFParameters(v_th=torch.as_tensor(0.4))
        self.debug = False
        self.use_encoder = use_encoder
        self.encoder = ConstantCurrentLIFEncoder(seq_length)

        ## Network Parameters

        # Scales the CNN output by a factor to improve the ConstantCurrentLIFEncoder Input
        self.cnn_scaler = 4
        self.encoder_scaler = 1

        self.cnn = C3NN_Feature_Extractor(debug= self.debug)
        self.snn = SNN_Model_b(n_classes, seq_length, p, uses_ts=use_encoder, debug= self.debug)

        self.decoder = decode

    def forward(self, x):
        
        x = self.cnn_scaler * self.cnn(x)
        if self.debug:
            #print(f"CNN {x[0][:20]}")
            print(f"CNN {x.shape}")
        if self.use_encoder:
            x = self.encoder_scaler * self.encoder(x)
        if self.debug:
            #print(f"Encoder {x[7][0][:20]}")
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


class C3SNN_ModelT_scaled(nn.Module):
    def __init__(self, n_classes, use_encoder = True):
        super(C3SNN_ModelT_scaled, self).__init__()
        
        seq_length = 32
        p = LIFParameters(v_th=torch.as_tensor(0.4))
        self.debug = False
        self.use_encoder = use_encoder
        self.encoder = ConstantCurrentLIFEncoder(seq_length)

        ## Network Parameters

        # Scales the CNN output by a factor to improve the ConstantCurrentLIFEncoder Input
        self.cnn_scaler = 5
        self.encoder_scaler = 2

        self.cnn = C3NN_Feature_Extractor(debug= self.debug)
        self.snn = SNN_Model_b(n_classes, seq_length, p, uses_ts=use_encoder, debug= self.debug)

        self.decoder = decode

    def forward(self, x):
        
        x = self.cnn_scaler * self.cnn(x)
        if self.debug:
            #print(f"CNN {x[0][:20]}")
            print(f"CNN {x.shape}")
        if self.use_encoder:
            x = self.encoder_scaler * self.encoder(x)
        if self.debug:
            #print(f"Encoder {x[7][0][:20]}")
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

class C3SNN_ModelT_No_Encoder(nn.Module):
    def __init__(self, n_classes, use_encoder = False):
        super(C3SNN_ModelT_No_Encoder, self).__init__()
        
        seq_length = 32
        p = LIFParameters(v_th=torch.as_tensor(0.4))
        self.debug = False
        self.use_encoder = use_encoder
        self.encoder = ConstantCurrentLIFEncoder(seq_length)

        ## Network Parameters

        # Scales the CNN output by a factor to improve the ConstantCurrentLIFEncoder Input
        self.cnn_scaler = 5
        self.encoder_scaler = 2

        self.cnn = C3NN_Feature_Extractor(debug= self.debug)
        self.snn = SNN_Model_b(n_classes, seq_length, p, uses_ts=use_encoder, debug= self.debug)

        self.decoder = decode

    def forward(self, x):
        
        x = self.cnn_scaler * self.cnn(x)
        if self.debug:
            #print(f"CNN {x[0][:20]}")
            print(f"CNN {x.shape}")
        
        if self.debug:
            #print(f"Encoder {x[7][0][:20]}")
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


class C3DSNN_ModelT(nn.Module):
    def __init__(self, n_classes, use_encoder = False):
        super(C3DSNN_ModelT, self).__init__()
        
        seq_length = 32
        p = LIFParameters(v_th=torch.as_tensor(0.4))
        self.debug = False
        self.use_encoder = use_encoder
        self.encoder = ConstantCurrentLIFEncoder(seq_length)

        ## Network Parameters

        # Scales the CNN output by a factor to improve the ConstantCurrentLIFEncoder Input
        self.cnn_scaler = 4
        self.encoder_scaler = 1

        self.cnn = C3DNN_Feature_Extractor(debug= self.debug)
        self.snn = SNN_Model_a(n_classes, seq_length, p, uses_ts=use_encoder, debug= self.debug)

        self.decoder = decode

    def forward(self, x):
        
        x = self.cnn_scaler * self.cnn(x)
        if self.debug:
            #print(f"CNN {x[0][:20]}")
            print(f"CNN {x.shape}")
        if self.use_encoder:
            x = self.encoder_scaler * self.encoder(x)
        if self.debug:
            #print(f"Encoder {x[7][0][:20]}")
            print(f"Encoder {x.shape}")
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

class C3DSNN_ModelT2(nn.Module):
    def __init__(self, n_classes, use_encoder = False):
        super(C3DSNN_ModelT2, self).__init__()
        
        seq_length = 48
        p = LIFParameters(v_th=torch.as_tensor(0.5))
        self.debug = False
        self.use_encoder = use_encoder
        self.encoder = ConstantCurrentLIFEncoder(seq_length)

        ## Network Parameters

        # Scales the CNN output by a factor to improve the ConstantCurrentLIFEncoder Input
        self.cnn_scaler = 7
        self.encoder_scaler = 1.5

        self.cnn = C3DNN_Feature_Extractor(debug= self.debug)
        self.snn = SNN_Model_a(n_classes, seq_length, p, uses_ts=use_encoder, debug= self.debug)

        self.decoder = decode

    def forward(self, x):
        
        x = self.cnn_scaler * self.cnn(x)
        if self.debug:
            #print(f"CNN {x[0][:20]}")
            print(f"CNN {x.shape}")
        if self.use_encoder:
            x = self.encoder_scaler * self.encoder(x)
        if self.debug:
            #print(f"Encoder {x[7][0][:20]}")
            print(f"Encoder {x.shape}")
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


class C3DSNN_C3D_ModelT(nn.Module):
    def __init__(self, n_classes):
        super(C3DSNN_C3D_ModelT, self).__init__()
        
        self.cnn_scaler = 4
        self.encoder_scaler = 1
        seq_length = 32
        p = LIFParameters(v_th=torch.as_tensor(0.4))
        self.debug = False
        self.use_encoder = True
        self.encoder = ConstantCurrentLIFEncoder(seq_length)

        self.cnn = C3D(2048, True)
        self.snn = SNN_Model_a(n_classes, seq_length, p, uses_ts=self.use_encoder, debug= self.debug)

        self.decoder = decode

    def forward(self, x):
        
        x = self.cnn_scaler * self.cnn(x)
        if self.debug:
            #print(f"CNN {x[0][:20]}")
            print(f"CNN {x.shape}")
        if self.use_encoder:
            x = self.encoder_scaler * self.encoder(x)
        if self.debug:
            #print(f"Encoder {x[7][0][:20]}")
            print(f"Encoder {x.shape}")
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






class C3DSNN_Fire_ModelT(nn.Module):
    def __init__(self, n_classes, use_encoder = False):
        super(C3DSNN_Fire_ModelT, self).__init__()
        
        seq_length = 32
        p = LIFParameters(v_th=torch.as_tensor(0.4))
        self.debug = False
        self.use_encoder = use_encoder
        self.encoder = ConstantCurrentLIFEncoder(seq_length)

        ## Network Parameters

        # Scales the CNN output by a factor to improve the ConstantCurrentLIFEncoder Input
        self.cnn_scaler = 4
        self.encoder_scaler = 1

        self.cnn = C3DNN_Small_t(debug= self.debug)
        self.snn = SNN_Model_a(n_classes, seq_length, p, uses_ts=use_encoder, debug= self.debug)

        self.decoder = decode

    def forward(self, x):
        
        x = self.cnn_scaler * self.cnn(x)
        if self.debug:
            #print(f"CNN {x[0][:20]}")
            print(f"CNN {x.shape}")
        if self.use_encoder:
            x = self.encoder_scaler * self.encoder(x)
        if self.debug:
            #print(f"Encoder {x[7][0][:20]}")
            print(f"Encoder {x.shape}")
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