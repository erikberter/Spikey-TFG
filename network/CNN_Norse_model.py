import torch
import torch.nn as nn

from norse.torch.module import SequentialState

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.lif import LIFRecurrentCell, LIFCell
from norse.torch.module.leaky_integrator import LILinearCell

import torchvision.models as models

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
