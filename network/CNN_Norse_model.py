import torch
import torch.nn as nn

from norse.torch.module import SequentialState

from norse.torch.functional.lif import LIFParameters, lif_current_encoder
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.lif import LIFRecurrentCell, LIFCell
from norse.torch.module.leaky_integrator import LILinearCell

from norse.torch.module.lsnn import LSNNCell

import torchvision.models as models




def spike_latency_lif_encode(
    input_current: torch.Tensor,
    seq_length: int,
    p: LIFParameters = LIFParameters(),
    dt=0.001,
) -> torch.Tensor:
    """Encodes an input value by the time the first spike occurs.
    Similar to the ConstantCurrentLIFEncoder, but the LIF can be
    thought to have an infinite refractory period.

    Parameters:
        input_current (torch.Tensor): Input current to encode (needs to be positive).
        sequence_length (int): Number of time steps in the resulting spike train.
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Integration time step (should coincide with the integration time step used in the model)
    """
    voltage = torch.zeros_like(input_current)
    z = torch.zeros_like(input_current)
    mask = torch.zeros_like(input_current)
    spikes = []
    

    for _ in range(seq_length):
        z, voltage = lif_current_encoder(
            input_current=input_current, voltage=voltage, p=p, dt=dt
        )

        spikes += [torch.where(mask > 0, torch.zeros_like(z), z)]
        mask += z

    return torch.stack(spikes)


class SpikeLatencyLIFEncoder(torch.nn.Module):
    """Encodes an input value by the time the first spike occurs.
    Similar to the ConstantCurrentLIFEncoder, but the LIF can be
    thought to have an infinite refractory period.
    Parameters:
        sequence_length (int): Number of time steps in the resulting spike train.
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Integration time step (should coincide with the integration time step used in the model)
    """

    def __init__(self, seq_length, p=LIFParameters(), dt=0.001):
        super(SpikeLatencyLIFEncoder, self).__init__()
        self.seq_length = seq_length
        self.p = p
        self.dt = dt

    def forward(self, input_current):
        return spike_latency_lif_encode(
            input_current, self.seq_length, self.p, self.dt
        )



class CNN_Norse_model(nn.Module):
    def __init__(self, n_classes):
        super(CNN_Norse_model, self).__init__()
        self.state_dim = 4
        self.input_features = 16
        self.hidden_features = 128
        self.output_features = 2
        self.seq_length = 2
        self.n_classes = n_classes
        self.constant_current_encoder = ConstantCurrentLIFEncoder(16)
        
        p=LIFParameters(v_th=torch.as_tensor(0.4))

        self.convs = [
            nn.Conv2d( 3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False)
        ]
        #for conv in self.convs:
        #    conv.weight.data.uniform_(-0.1, 0.3)

        self.features = SequentialState(
            # preparation
            # Dimension [3, 112, 112]
            nn.BatchNorm2d(3),
            self.convs[0], 
            
        )

        self.features2 = SequentialState(
            nn.ReLU(),
            # Dimension [8, 55, 55] ~ 96800
            nn.BatchNorm2d(8),
            self.convs[1],
            nn.ReLU(),
             # Dimension [16, 27, 27] ~ 186624
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
             # Dimension [16, 13, 13] ~ 48400
            self.convs[2],
            nn.ReLU(),
             # Dimension [32, 26, 26] ~ 96800
            nn.BatchNorm2d(24),
            nn.MaxPool2d(2),
            
             # Dimension [32, 27, 27] ~ 23328
            self.convs[3],
            nn.ReLU(),
             # Dimension [32, 26, 26] ~ 21632
            torch.nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
             # Dimension [32, 13, 13] ~ 5408
            self.convs[4],
            nn.ReLU(),
            #LIFCell(p),
             # Dimension [64, 7, 7] ~ 3136
            torch.nn.BatchNorm2d(48),
            nn.Flatten(),
        )

        self.classification = SequentialState(
            # Classification
            #LIFCell(p),
            
            nn.Linear(768, 256, bias=False),
            #LIFCell(p),
            nn.ReLU(),
            nn.Linear(256, n_classes, bias=False),
        )
        
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        
        voltages = torch.empty(
            self.seq_length, x.shape[0], self.n_classes, device=x.device, dtype=x.dtype
        )
        sf = None
        sc = None
        #print(f"Casca0 {x[0][0]}")
        for ts in range(self.seq_length):
            out_f, sf = self.features(x, sf)
            #print(f"Cosca {self.features[0].weight.data}")
            #print(f"Casca {out_f[0][0]}")

            out_f2, sc1 = self.features2(out_f, sf)
            #print(f"Casca1 {out_f2}")

            out_c, sc = self.classification(out_f2, sc1)
            #print(f"Casca2 {out_c}")
            
            #input("POPO")
            voltages[ts, :, :] = out_c + 0.001 * torch.randn(
                x.shape[0], self.n_classes, device=x.device
            )
        
        y_hat, _ = torch.max(voltages, 0)
        
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        
        return y_hat



class ResNet_SNN(nn.Module):
    def __init__(self, n_classes):
        super(ResNet_SNN, self).__init__()
        self.state_dim = 4
        self.input_features = 16
        self.hidden_features = 128
        self.output_features = 2
        self.seq_length = 24
        self.n_classes = n_classes
        self.uses_ts = False
        self.constant_current_encoder = ConstantCurrentLIFEncoder(self.seq_length)
        
        p=LIFParameters(v_th=torch.as_tensor(0.33))

        self.features = models.video.r3d_18(pretrained=True)
        for param in self.features.parameters():
            param.requires_grad = False

        self.classification = SequentialState(
            
            nn.Linear(400, 128, bias=False),
            LIFCell(p),
            nn.Dropout(0.2),
            nn.Linear(128, 128, bias=False),
            LIFCell(p),
            nn.Dropout(0.2),
            nn.Linear(128, 128, bias=False),
            LILinearCell(128, n_classes, p),
        )

        self.relu = nn.ReLU()
        
        self.saved_log_probs = []
        self.rewards = []

        self.feature_scalar = torch.nn.Parameter(torch.ones(1)) 
        self.encoder_scalar = torch.nn.Parameter(torch.ones(1))

    def print_params(self):
        print(f"Feature Scalar {self.feature_scalar}")
        print(f"Encoder Scalar {self.encoder_scalar}")

    def forward(self, x):
        batch_size = x.shape[1] if self.uses_ts else x.shape[0]
        voltages = torch.empty(
            self.seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
        )

        out = 2 * self.feature_scalar * self.features(x)
        out_f = torch.flatten(out, start_dim=1)
        #out_f = self.relu(out_f)
        #print(f"La media en feature es de {torch.mean(out_f)}")
        #input("Pausa")
        encoded = 5 * self.encoder_scalar * self.constant_current_encoder(out_f)

        sc = None
        for ts in range(self.seq_length):
            z = encoded[ts, :] 
            #print(f"La media en {ts} es de {torch.mean(z)}")
            #input("Pausa")
            out_c, sc = self.classification(z, sc)
            voltages[ts, :, :] = out_c
        
        y_hat = voltages[-1]
        
        return y_hat


class ResNet_SNN_exp(nn.Module):
    def __init__(self, n_classes):
        super(ResNet_SNN_exp, self).__init__()

        self.seq_length = 32
        self.n_classes = n_classes
        self.uses_ts = False
        self.constant_current_encoder = ConstantCurrentLIFEncoder(self.seq_length)
        
        p=LIFParameters(v_th=torch.as_tensor(0.33))

        self.features = models.video.r3d_18(pretrained=True)
        #for param in self.features.parameters():
        #    param.requires_grad = False
        self.classification = SequentialState(
            
            nn.Linear(400, 128, bias=False),
            LIFCell(p),
            nn.Dropout(0.25),
            nn.Linear(128, 128, bias=False),
            LIFCell(p),
            nn.Dropout(0.25),
            nn.Linear(128, 128, bias=False),
            LIFCell(p),
            nn.Dropout(0.25),
            LILinearCell(128, n_classes, p),
        )

        self.batchnorm = nn.BatchNorm1d(400)
        

        self.feature_scalar = torch.nn.Parameter(torch.ones(1)) 
        self.encoder_scalar = torch.nn.Parameter(torch.ones(1))

    def print_params(self):
        print(f"Feature Scalar {self.feature_scalar}")
        print(f"Encoder Scalar {self.encoder_scalar}")

    def forward(self, x):
        batch_size = x.shape[1] if self.uses_ts else x.shape[0]
        voltages = torch.empty(
            self.seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
        )

        out = self.features(x)
        #out = self.batchnorm(out)
        #print(f"El out_0 es de {out[0]}")
        #input("Pausa")
        out_f = torch.flatten(out, start_dim=1)
        #out_f =  * self.relu(out_f)
        
        
        print(f"El out_f es de {out_f[0]}")
        encoded = 5 *  self.constant_current_encoder( 2 * out_f)

        sc, sl0,sl1,sl2 = None, None,None,None
        for ts in range(self.seq_length):
            z = encoded[ts, :] 
            #print(f"La media en {ts} es de {torch.mean(z)}")
            #input("Pausa")
            out_c, sc = self.classification(z, sc)
            voltages[ts, :, :] = out_c
        
        y_hat = voltages[-1]
        print(f"El y_hat es de {y_hat[0]}")
        
        return y_hat


class ResNet_SNN_expV(nn.Module):
    def __init__(self, n_classes):
        super(ResNet_SNN_expV, self).__init__()

        self.seq_length = 24
        self.n_classes = n_classes
        self.uses_ts = False
        self.constant_current_encoder = ConstantCurrentLIFEncoder(self.seq_length)
        
        p=LIFParameters(v_th=torch.as_tensor(0.23))

        self.features = models.video.r3d_18(pretrained=True)
        #for param in self.features.parameters():
        #    param.requires_grad = False
        
            
        self.lin =     nn.Linear(400, 128, bias=True)
        self.lif =    LIFCell(p)
        self.drop =    nn.Dropout(0.25)
        self.lin1 =    nn.Linear(128, 128, bias=True)
        self.lif1 =    LIFCell(p)
        self.drop1 =    nn.Dropout(0.25)
        self.lin2 =    nn.Linear(128, 128, bias=True)
        self.lif2 =    LIFCell(p)
        self.drop2 =    nn.Dropout(0.25)
        self.li =    LILinearCell(128, n_classes, p)
        

        self.batchnorm = nn.BatchNorm1d(400)
        

        self.feature_scalar = torch.nn.Parameter(torch.ones(1)) 
        self.encoder_scalar = torch.nn.Parameter(torch.ones(1))

    def print_params(self):
        print(f"Feature Scalar {self.feature_scalar}")
        print(f"Encoder Scalar {self.encoder_scalar}")

    def forward(self, x):
        batch_size = x.shape[1] if self.uses_ts else x.shape[0]
        voltages = torch.empty(
            self.seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
        )

        out = self.features(x)
        #out = self.batchnorm(out)
        #print(f"El out_0 es de {out[0]}")
        #input("Pausa")
        out_f = torch.flatten(out, start_dim=1)
        #out_f =  * self.relu(out_f)
        
        
        #print(f"El out_f es de {out_f[0]}")
        encoded = self.constant_current_encoder(out_f)

        sc, sl0,sl1,sl2 = None, None,None,None
        for ts in range(self.seq_length):
            z = encoded[ts, :] 
            #print(f"La media en {ts} es de {torch.mean(z)}")
            #input("Pausa")
            #out_c, sc = self.classification(z, sc)

            out = self.lin(z)
            out, sl0 = self.lif(out, sl0) 
            out = self.drop(out)

            out_1 = self.lin1( out)
            out_1, sl1 = self.lif1(out_1, sl1)
            out_1 = self.drop1(out_1)

            out_1 = out_1 + out

            out_2 = self.lin2( out_1)
            out_2, sl2 = self.lif2(out_2, sl2)
            out_2 = self.drop2(out_2)
            
            out_2 = out_2 + out_1

            out_f, sc = self.li(out_2, sc)


            voltages[ts, :, :] = out_f
        
        y_hat = voltages[-1]
        #print(f"El y_hat es de {y_hat[0]}")
        
        return y_hat


class ResNet_SNN_expVSL(nn.Module):
    def __init__(self, n_classes):
        super(ResNet_SNN_expVSL, self).__init__()

        self.seq_length = 24
        self.n_classes = n_classes
        self.uses_ts = False
        self.constant_current_encoder = SpikeLatencyLIFEncoder(self.seq_length)
        
        p=LIFParameters(v_th=torch.as_tensor(0.085))

        self.features = models.video.r3d_18(pretrained=True)
        #for param in self.features.parameters():
        #    param.requires_grad = False
        
            
        self.lin =     nn.Linear(400, 128, bias=False)
        self.lif =    LIFCell(p)
        self.drop =    nn.Dropout(0.25)
        self.lin1 =    nn.Linear(128, 128, bias=False)
        self.lif1 =    LIFCell(p)
        self.drop1 =    nn.Dropout(0.25)
        self.lin2 =    nn.Linear(128, 128, bias=False)
        self.lif2 =    LIFCell(p)
        self.drop2 =    nn.Dropout(0.25)
        self.li =    LILinearCell(128, n_classes, p)
        

        self.batchnorm = nn.BatchNorm1d(400)
        

        self.feature_scalar = torch.nn.Parameter(torch.ones(1)) 
        self.encoder_scalar = torch.nn.Parameter(torch.ones(1))

    def print_params(self):
        print(f"Feature Scalar {self.feature_scalar}")
        print(f"Encoder Scalar {self.encoder_scalar}")

    def forward(self, x):
        batch_size = x.shape[1] if self.uses_ts else x.shape[0]
        voltages = torch.empty(
            self.seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
        )

        out = self.features(x)
        #out = self.batchnorm(out)
        #print(f"El out_0 es de {out[0]}")
        #input("Pausa")
        out_f = torch.flatten(out, start_dim=1)
        #out_f =  * self.relu(out_f)
        
        
        #print(f"El out_f es de {out_f[0]}")
        encoded = self.constant_current_encoder(out_f)

        sc, sl0,sl1,sl2 = None, None,None,None
        for ts in range(self.seq_length):
            z = encoded[ts, :] 
            #print(f"La media en {ts} es de {torch.mean(z)}")
            #input("Pausa")
            #out_c, sc = self.classification(z, sc)

            out = self.lin(z)
            out, sl0 = self.lif(out, sl0) 
            out = self.drop(out)

            out_1 = self.lin1( out)
            out_1, sl1 = self.lif1(out_1, sl1)
            out_1 = self.drop1(out_1)

            out_1 = out_1 + out

            out_2 = self.lin2( out_1)
            out_2, sl2 = self.lif2(out_2, sl2)
            out_2 = self.drop2(out_2)
            
            out_2 = out_2 + out_1

            out_f, sc = self.li(out_2, sc)


            voltages[ts, :, :] = out_f
        
        
        #y_hat = voltages[-1]
        y_hat, _ = torch.max(voltages, 0)

        #print("----------------------")

        #print(f"y_hat is {y_hat}")

        #input("Stop")
        #print(f"El y_hat es de {y_hat[0]}")
        
        return y_hat



class ResNet_SNN_InverseScale(nn.Module):
    def __init__(self, n_classes):
        super(ResNet_SNN_InverseScale, self).__init__()
        self.state_dim = 4
        self.input_features = 16
        self.hidden_features = 128
        self.output_features = 2
        self.seq_length = 24
        self.n_classes = n_classes
        self.uses_ts = False
        self.constant_current_encoder = ConstantCurrentLIFEncoder(self.seq_length)
        
        p=LIFParameters(v_th=torch.as_tensor(0.33))

        self.features = models.video.r3d_18(pretrained=True)
        for param in self.features.parameters():
            param.requires_grad = False

        self.classification = SequentialState(
            
            nn.Linear(400, 128, bias=False),
            LIFCell(p),
            nn.Dropout(0.2),
            nn.Linear(128, 128, bias=False),
            LIFCell(p),
            nn.Dropout(0.2),
            nn.Linear(128, 128, bias=False),
            LILinearCell(128, n_classes, p),
        )

        self.relu = nn.ReLU()
        
        self.saved_log_probs = []
        self.rewards = []

        self.feature_scalar = torch.nn.Parameter(torch.ones(1)) 
        self.encoder_scalar = torch.nn.Parameter(torch.ones(1))

    def print_params(self):
        print(f"Feature Scalar {self.feature_scalar}")
        print(f"Encoder Scalar {self.encoder_scalar}")

    def forward(self, x):
        batch_size = x.shape[1] if self.uses_ts else x.shape[0]
        voltages = torch.empty(
            self.seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
        )

        out = 5 * self.feature_scalar * self.features(x)
        out_f = torch.flatten(out, start_dim=1)
        #out_f = self.relu(out_f)
        #print(f"La media en feature es de {torch.mean(out_f)}")
        #input("Pausa")
        encoded = 2 * self.encoder_scalar * self.constant_current_encoder(out_f)

        sc = None
        for ts in range(self.seq_length):
            z = encoded[ts, :] 
            #print(f"La media en {ts} es de {torch.mean(z)}")
            #input("Pausa")
            out_c, sc = self.classification(z, sc)
            voltages[ts, :, :] = out_c
        
        y_hat = voltages[-1]
        
        return y_hat




class ResNet_Cop_SNN(nn.Module):
    def __init__(self, n_classes):
        super(ResNet_Cop_SNN, self).__init__()
        self.state_dim = 4
        self.input_features = 16
        self.hidden_features = 128
        self.output_features = 2
        self.seq_length = 24
        self.n_classes = n_classes
        self.uses_ts = False
        self.constant_current_encoder = ConstantCurrentLIFEncoder(self.seq_length)
        
        p=LIFParameters(v_th=torch.as_tensor(0.33))

        self.features = models.video.r3d_18(pretrained=True)
        for param in self.features.parameters():
            param.requires_grad = False

        self.classification = SequentialState(
            
            nn.Linear(400, 128, bias=False),
            LIFCell(p),
            nn.Dropout(0.25),
            nn.Linear(128, 128, bias=False),
            LIFCell(p),
            nn.Dropout(0.25),
            nn.Linear(128, 128, bias=False),
            LIFCell(p),
            nn.Dropout(0.25),
            LILinearCell(128, n_classes, p),
        )

        self.relu = nn.ReLU()
        
        self.saved_log_probs = []
        self.rewards = []

        self.feature_scalar = torch.nn.Parameter(torch.ones(1)) 
        self.encoder_scalar = torch.nn.Parameter(torch.ones(1))

    def print_params(self):
        print(f"Feature Scalar {self.feature_scalar}")
        print(f"Encoder Scalar {self.encoder_scalar}")

    def forward(self, x):
        batch_size = x.shape[1] if self.uses_ts else x.shape[0]
        voltages = torch.empty(
            self.seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
        )

        out = 2 * self.feature_scalar * self.features(x)
        out_f = torch.flatten(out, start_dim=1)
        #out_f = self.relu(out_f)
        #print(f"La media en feature es de {torch.mean(out_f)}")
        #input("Pausa")
        encoded = 5 * self.encoder_scalar * self.constant_current_encoder(out_f)

        sc = None
        for ts in range(self.seq_length):
            z = encoded[ts, :] 
            #print(f"La media en {ts} es de {torch.mean(z)}")
            #input("Pausa")
            out_c, sc = self.classification(z, sc)
            voltages[ts, :, :] = out_c
        
        y_hat = voltages[-1]
        
        return y_hat


class SNN_Cont(nn.Module):
    def __init__(self, n_classes):
        super(SNN_Cont, self).__init__()
        self.n_classes = n_classes
        self.seq_length = 16
        self.constant_current_encoder = ConstantCurrentLIFEncoder(self.seq_length)
        
        p=LIFParameters(v_th=torch.as_tensor(0.33))

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size= 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(1,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),

            nn.MaxPool3d(2),

            nn.Conv3d(64, 64, kernel_size=(1,3,3), stride=(1,2,2), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(1,3,3), stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool3d((1,2,2)),
        )

        self.classification = SequentialState(
            
            nn.Linear(576, 128, bias=False),
            LSNNCell(),
            nn.Dropout(0.2),
            nn.Linear(128, 128, bias=False),
            LSNNCell(),
            nn.Dropout(0.2),
            nn.Linear(128, 128, bias=False),
            LSNNCell(),
            nn.Dropout(0.2),
            LILinearCell(128, n_classes, p),
        )

        self.relu = nn.ReLU()
        
        self.feature_scalar = torch.nn.Parameter(torch.ones(1)) 
        self.encoder_scalar = torch.nn.Parameter(torch.ones(1))

    def print_params(self):
        print(f"Feature Scalar {self.feature_scalar}")
        print(f"Encoder Scalar {self.encoder_scalar}")

    def forward(self, x):
        # [batch, ts, cha, img, img]


        batch_size = x.shape[0]
        voltages = torch.empty(
            self.seq_length, batch_size, self.n_classes, device=x.device, dtype=x.dtype
        )
        #print(x.shape)

        out = 11 * self.feature_scalar * self.features(x)
        #print(f"La out shape es {out.shape}")
        #out_f = torch.flatten(out, start_dim=1)
        #out_f = self.relu(out_f)
        #print(f"La media en feature es de {torch.mean(out)}")
        #input("Pausa")
        encoded = 8 * self.encoder_scalar * self.constant_current_encoder(out)
        # [batch, sql, ts, img, img]

        #print(f"La encde  shape es {encoded.shape}")
        #input("Pause")
        encoded = encoded.reshape(encoded.shape[0]*encoded.shape[3], encoded.shape[1], encoded.shape[2],encoded.shape[4],encoded.shape[5])
        #print(f"La encde  shape2 es {encoded.shape}")
        #input("Pause")
        sc = None
        for ts in range(self.seq_length):
            z = encoded[ts, :] 
            z = torch.flatten(z, start_dim=1)
            #print(f"La media en feature es de {torch.mean(z)}")
            #input(f"Pausa {ts}")
            out_c, sc = self.classification(z, sc)
            voltages[ts, :, :] = out_c
        
        #print(voltages[-1])
        #input("STOP")
        y_hat = voltages[-1]
        
        return y_hat