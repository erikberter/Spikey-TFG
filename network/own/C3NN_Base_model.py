import torch.nn as nn


class C3DNN(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, n_classes,  debug = False):
        super(C3DNN, self).__init__()

        self.debug = debug

        self.features = nn.Sequential(
            nn.BatchNorm3d(3),
            nn.Conv3d(3, 32, kernel_size=(3,5,5), stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(64),

            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 128, kernel_size=(1,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            
            nn.MaxPool3d(2),

            nn.Flatten(),
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )



    def forward(self, x):
        return self.features(x)


class C3DNN_Small(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, n_classes,  debug = False):
        super(C3DNN_Small, self).__init__()

        self.debug = debug

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size= 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),

            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(1,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            
            nn.MaxPool3d(2),

            nn.Flatten(),
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )



    def forward(self, x):
        return self.features(x)

class C3DNN_NB_Small(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, n_classes,  debug = False):
        super(C3DNN_NB_Small, self).__init__()

        self.debug = debug

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size= 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm3d(64),
            
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm3d(64),

            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(1,3,3), stride=1, padding=1),
            nn.ReLU(),
            #nn.BatchNorm3d(128),
            
            nn.MaxPool3d(2),

            nn.Flatten(),
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )



    def forward(self, x):
        return self.features(x)




class C3DNN_Small_Alt(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, n_classes,  debug = False):
        super(C3DNN_Small_Alt, self).__init__()

        self.debug = debug

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size= 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),

            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 128, kernel_size=(1,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            
            nn.MaxPool3d(2),

            nn.Flatten(),
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )



    def forward(self, x):
        return self.features(x)

class C3DNN_Med_Alt(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, n_classes,  debug = False):
        super(C3DNN_Med_Alt, self).__init__()

        self.debug = debug

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size= 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),

            nn.MaxPool3d(2),

            nn.Conv3d(128, 128, kernel_size=(3,3,3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, kernel_size=(1,3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(256),
            
            nn.MaxPool3d(2),

            nn.Flatten(),
            nn.Linear(1152*2, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )



    def forward(self, x):
        return self.features(x)

class C3DNN_NB(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, n_classes,  debug = False):
        super(C3DNN_NB, self).__init__()

        self.debug = debug

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3,3,3)),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2), stride=(1,2,2)),

            nn.Conv3d(32, 64, kernel_size=(3,3,3)),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2), stride=(1,2,2)),

            nn.Conv3d(64, 64, kernel_size=(3,3,3)),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2), stride=(1,2,2)),

            nn.Conv3d(64, 128, kernel_size=(3,3,3)),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3,3,3)),
            nn.ReLU(),

            nn.MaxPool3d((1,2,2), stride=(1,2,2)),

            nn.Conv3d(128, 256, kernel_size=(2,2,2)),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=(2,2,2)),
            nn.ReLU(),

            nn.MaxPool3d((1,2,2), stride=(1,2,2)),


            nn.Flatten(),

            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, n_classes),
        )



    def forward(self, x):
        return self.features(x)
    