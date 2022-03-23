import torch.nn as nn
import torch

class TimeDistributed(nn.Module):
    "Applies a module over tdim identically for each step" 
    def __init__(self, module, low_mem=False, tdim=1):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim
        
    def forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim!=1: 
            return self.low_mem_forward(*args)
        else:
            #only support tdim=1
            inp_shape = args[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]   
            out = self.module(*[x.reshape(bs*seq_len, *x.shape[2:]) for x in args], **kwargs)
            out_shape = out.shape
            #print(f"Act shape {out_shape}")
            pepe = out.reshape(bs, seq_len,*out_shape[1:])
            #print(f"Act2 shape {pepe.shape}")
            return pepe
    
    def low_mem_forward(self, *args, **kwargs):                                           
        "input x with shape:(bs,seq_len,channels,width,height)"
        tlen = args[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in args]
        out = []
        for i in range(tlen):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        return torch.stack(out,dim=self.tdim)

    def __repr__(self):
        return f'TimeDistributed({self.module})'

class CNN_LSTM(nn.Module):
    """
        Conv2D NN Model to extract features from image
        LSTM to clasiffy
    """


    def __init__(self, n_classes, timesteps,  debug = False):
        super(CNN_LSTM, self).__init__()

        self.debug = debug

        self.features = nn.Sequential(
            TimeDistributed(nn.Conv2d(3, 32, 3, 2, 1), tdim=timesteps),
            nn.ReLU(),
            TimeDistributed(nn.Conv2d(32, 64, 3, 1, 1), tdim=timesteps),
            nn.ReLU(),
            TimeDistributed(nn.BatchNorm2d(64), tdim=timesteps),
            
            TimeDistributed(nn.MaxPool2d(2), tdim=timesteps),
            

            TimeDistributed(nn.Conv2d(64, 64, 3, 1, 1), tdim=timesteps),
            TimeDistributed(nn.BatchNorm2d(64), tdim=timesteps),

            TimeDistributed(nn.MaxPool2d(2), tdim=timesteps),

            TimeDistributed(nn.Conv2d(64, 64, 3, 1, 1), tdim=timesteps),
            nn.ReLU(),
            TimeDistributed(nn.Conv2d(64, 128, 3, 1, 1), tdim=timesteps),
            nn.ReLU(),
            TimeDistributed(nn.BatchNorm2d(128), tdim=timesteps),
            


            TimeDistributed(nn.Conv2d(128, 128, 3, 2, 1), tdim=timesteps),
            nn.ReLU(),
            TimeDistributed(nn.Conv2d(128, 256, 3, 1, 1), tdim=timesteps),
            nn.ReLU(),
            TimeDistributed(nn.BatchNorm2d(256), tdim=timesteps),
            
            TimeDistributed(nn.MaxPool2d(2), tdim=timesteps),

            TimeDistributed(nn.Flatten(), tdim=timesteps),
            
        )
        self.lstm = nn.LSTM(256*3*3, 256, 2, batch_first=True)

        self.classification = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )


    def forward(self, x):
        out =  self.features(x)
        out, (hn,cn) = self.lstm(out)
        out = self.classification(hn[-1])
        return out


class CNN_LSTM_Alt(nn.Module):
    """
        Conv2D NN Model to extract features from image
        LSTM to clasiffy

        Bigger LSTM and bigger classification
    """


    def __init__(self, n_classes, t_dim = 1,  debug = False):
        super(CNN_LSTM_Alt, self).__init__()

        self.debug = debug

        self.features = nn.Sequential(
            TimeDistributed(nn.Conv2d(3, 32, 3, 2, 1), tdim=t_dim),
            nn.ReLU(),
            TimeDistributed(nn.Conv2d(32, 64, 3, 1, 1), tdim=t_dim),
            nn.ReLU(),
            TimeDistributed(nn.BatchNorm2d(64), tdim=t_dim),
            
            TimeDistributed(nn.MaxPool2d(2), tdim=t_dim),
            
            TimeDistributed(nn.Conv2d(64, 64, 3, 1, 1), tdim=t_dim),
            nn.ReLU(),
            TimeDistributed(nn.Conv2d(64, 64, 3, 1, 1), tdim=t_dim),
            nn.ReLU(),
            TimeDistributed(nn.BatchNorm2d(64), tdim=t_dim),

            
            TimeDistributed(nn.MaxPool2d(2), tdim=t_dim),

            TimeDistributed(nn.Conv2d(64, 128, 3, 2, 1), tdim=t_dim),
            nn.ReLU(),
            TimeDistributed(nn.Conv2d(128, 128, 3, 1, 1), tdim=t_dim),
            nn.ReLU(),
            TimeDistributed(nn.BatchNorm2d(128), tdim=t_dim),

            
            TimeDistributed(nn.MaxPool2d(2), tdim=t_dim),

            TimeDistributed(nn.Flatten(), tdim=t_dim),
            
        )
        self.lstm = nn.LSTM(128*3*3, 512, 2, batch_first=True, bidirectional  = True)

        self.classification = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )


    def forward(self, x):
        out =  self.features(x)
        out, (hn,cn) = self.lstm(out)
        out = self.classification(hn[-1])
        return out
