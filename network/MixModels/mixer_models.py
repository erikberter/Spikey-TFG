import torch
import torch.nn as nn

import torchvision.models as models

from network.own.C3NN_Base_model import C3DNN_Small

class MixClassification(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, n_classes,  debug = False):
        super(MixClassification, self).__init__()

        self.debug = debug

        self.classification = nn.Sequential(
            nn.Linear(2*n_classes, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )



    def forward(self, x):
        return self.classification(x)


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


class MixModel(nn.Module):
    """
        Mixer model composed of spatial, temporal and fusion models
    """

    def __init__(self, spatial, temporal, fusion,  debug = False):
        super(MixModel, self).__init__()

        self.debug = debug
        
        self.spatial = spatial
        self.temporal = temporal
        self.fusion = fusion



    def forward(self, x_spa, x_flow):

        x_spa = self.spatial(x_spa)
        x_temp = self.temporal(x_flow)

        x_res = torch.cat((x_spa, x_temp), 1)
        return self.fusion(x_res)



class MixModelDefault(nn.Module):
    """
        Mixer model composed of spatial, temporal and fusion models
    """

    def __init__(self, num_classes, debug = False):
        super(MixModelDefault, self).__init__()

        self.debug = debug

        self.spatial =  models.video.r3d_18(pretrained=True)
        self.temporal =  models.video.r3d_18(pretrained=True)
        self.fusion = MixClassificationBig(400,num_classes)



    def forward(self, x):

        x_spa = self.spatial(x[0])
        x_temp = self.temporal(x[1])

        x_res = torch.cat((x_spa, x_temp), 1)
        return self.fusion(x_res)

class MixModelSmall(nn.Module):
    """
        Mixer model composed of spatial, temporal and fusion models
    """

    def __init__(self, num_classes, debug = False):
        super(MixModelDefault, self).__init__()

        self.debug = debug

        self.spatial =  C3DNN_Small(num_classes)
        self.temporal =  C3DNN_Small(num_classes)
        self.fusion = MixClassificationBig(num_classes,num_classes)



    def forward(self, x):

        x_spa = self.spatial(x[0])
        x_temp = self.temporal(x[1])

        x_res = torch.cat((x_spa, x_temp), 1)
        return self.fusion(x_res)