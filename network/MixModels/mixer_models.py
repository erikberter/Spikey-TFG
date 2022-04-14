import torch
import torch.nn as nn

import torchvision.models as models

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



class C3NN_Small_Mix(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, n_classes,  debug = False):
        super(C3NN_Small_Mix, self).__init__()

        self.debug = debug

        self.bnpre = nn.BatchNorm3d(3)

        self.conv1 = nn.Conv3d(3, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = nn.BatchNorm3d(32)
        

        self.conv2_1 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm3d(32)

        
        self.conv3_1 = nn.Conv3d(32, 32, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.conv3_int = nn.Conv3d(32, 64, kernel_size=3, padding=1)

        self.bn3 = nn.BatchNorm3d(32)

        self.conv4_1 = nn.Conv3d(64, 64, kernel_size=1, padding=0)
        self.conv4_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.bn4 = nn.BatchNorm3d(64)

        
        self.conv5_1 = nn.Conv3d(64, 64, kernel_size=1, padding=0)
        self.conv5_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        
        self.conv5_int = nn.Conv3d(64, 128, kernel_size=3, padding=1)

        self.bn5 = nn.BatchNorm3d(64)

        self.conv6_1 = nn.Conv3d(128, 128, kernel_size=1, padding=0)
        self.conv6_2 = nn.Conv3d(128, 256, kernel_size=(1,3,3), padding=1)
        self.conv6_3 = nn.Conv3d(256, 256, kernel_size=(2,3,3), padding=0)

        self.bn6 = nn.BatchNorm3d(256)

        self.maxpool = nn.MaxPool3d(kernel_size=(2,3,3), padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2,3,3), padding=1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2,3,3), padding=1)
        self.maxpool1x3x3 = nn.MaxPool3d(kernel_size=(1,3,3),padding=1)

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        x = self.bnpre(x) # [16, 3, 171, 112]

        x = self.conv1(x) # [16, 32, 171, 112]
        x = self.bn1(x) 
        x = self.relu(x)
        x_1 = self.maxpool(x) # [8, 32, 57, 37]

        x_2 = self.conv2_1(x_1) # [8, 32, 57, 37]
        x_2 = self.conv2_2(x_2) 
        x_2 = self.bn2(x_2) 
        x_2 = self.relu(x_2)

        x_2 = x_2 + x_1

        x_3 = self.conv3_1(x_2) # [8, 32, 57, 37]
        x_3 = self.conv3_2(x_3)  # [8, 32, 57, 37]
        x_3 = self.bn3(x_3) 
        x_3 = self.relu(x_3)

        x_3 = x_2 + x_3

        x_3 = self.maxpool2(x_3) # [4, 32, 19, 12]

        x_3 = self.conv3_int(x_3)  # [8, 64, 57, 37]

        x_4 = self.conv4_1(x_3) # [4, 64, 19, 12]
        x_4 = self.conv4_2(x_4) 
        x_4 = self.bn4(x_4) 
        x_4 = self.relu(x_4)

        x_4 = x_4 + x_3

        x_5 = self.conv5_1(x_4) # [4, 64, 19, 12]
        x_5 = self.conv5_2(x_5)  # [4, 64, 19, 12]
        x_5 = self.bn5(x_5) 
        x_5 = self.relu(x_5)

        x_5 = x_5 + x_4

        x_6 = self.maxpool2(x_5) # [2, 64, 6, 4]

        x_6 = self.conv5_int(x_6)  # [2, 128, 6, 4]

        x_6 = self.conv6_1(x_6) # [2, 128, 6, 4]
        x_6 = self.conv6_2(x_6) # [2, 128, 6, 4]
        x_6 = self.conv6_3(x_6) # [1, 256, 3, 1]

        x_6 = self.bn6(x_6)
        x_6 = self.relu(x_6)

        

        x_6 = torch.flatten(x_6, start_dim=1)

        return x_6


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
