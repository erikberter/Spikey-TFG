import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate






class C3SNN_SNNT_ModelT(nn.Module):
    def __init__(self, n_classes):
        super(C3SNN_SNNT_ModelT, self).__init__()
        
        self.debug = False

        ## Network Parameters

        self.num_steps = 25 # number of time steps
        self.beta = 0.5  # neuron decay rate
        self.spike_grad = surrogate.fast_sigmoid()

        self.net = nn.Sequential(

            nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad),
            nn.MaxPool3d(2),
            nn.Conv3d(48, 64, kernel_size=3, stride=1, padding=1, bias=False),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad),

            nn.Flatten(),
            nn.Linear(1024, 256),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad, output=True),
            nn.Linear(256, 256),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad, output=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # Required data in shape (num_steps, batch_size, 1, 28, 28)

        
        # Data shape is now [batch_size, channels, timestep, height, width]
        #x = torch.transpose(x, 0, 2)
        #x = torch.transpose(x, 1, 2)


        _, o = self.net(x)
        return o