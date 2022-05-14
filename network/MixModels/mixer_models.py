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


class MixClassificationBig_expV(nn.Module):
    """
        Conv3D NN Model to extract features from image
    """


    def __init__(self, input_size, n_classes, debug = False):
        super(MixClassificationBig_expV, self).__init__()

        self.debug = debug


        self.lin =  nn.Linear(2 * input_size, 256, bias=True)
        self.relu =     nn.ReLU()
        self.drop =    nn.Dropout(0.25)
        self.lin1 =    nn.Linear(256, 256, bias=True)
        self.relu1 =    nn.ReLU()
        self.drop1 =    nn.Dropout(0.25)
        self.lin2=    nn.Linear(256, 256, bias=True)
        self.relu2 =     nn.ReLU()
        self.drop2 =    nn.Dropout(0.25)
        self.li =     nn.Linear(256, n_classes, bias=False)
        

    def forward(self, x):

        out = self.lin(x)
        out = self.relu(out)
        out = self.drop(out)

        out_1 = self.lin1(out)
        out_1 = self.relu1(out_1)
        out_1 = self.drop1(out_1)

        out_1 = out_1 + out

        out_2 = self.lin2(out_1)
        out_2 = self.relu2(out_2)
        out_2 = self.drop2(out_2)

        out_2 = out_2 + out_1

        out_f = self.li(out_2)

        return out_f















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

        for param in self.spatial.parameters():
            param.requires_grad = False

        for param in self.temporal.parameters():
            param.requires_grad = False


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


class MixModelDefault_expV(nn.Module):
    """
        Mixer model composed of spatial, temporal and fusion models
    """

    def __init__(self, num_classes, debug = False):
        super(MixModelDefault_expV, self).__init__()

        self.debug = debug

        self.spatial =  models.video.r3d_18(pretrained=True)
        self.temporal =  models.video.r3d_18(pretrained=True)
        self.fusion = MixClassificationBig_expV(400,num_classes)

        self.relu =  nn.ReLU()

    def forward(self, x):

        x_spa = self.spatial(x[0])
        x_temp = self.temporal(x[1])

        x_res = torch.cat((x_spa, x_temp), 1)


        return self.fusion(x_res)



class Minimi(nn.Module):
    def __init__(self, n_classes):
        super(Minimi, self).__init__()
        
        self.n_classes = n_classes
        self.uses_ts = False

        self.lin =  nn.Linear(400, 128, bias=True)
        self.relu =     nn.ReLU()
        self.drop =    nn.Dropout(0.25)
        self.lin1 =    nn.Linear(128, 128, bias=True)
        self.relu1 =    nn.ReLU()
        self.drop1 =    nn.Dropout(0.25)
        self.lin2=    nn.Linear(128, 128, bias=True)
        self.relu2 =     nn.ReLU()
        self.drop2 =    nn.Dropout(0.25)
        self.li =     nn.Linear(128, n_classes, bias=True)
        
        

    def forward(self, x):

        out = self.lin(x)
        out = self.relu(out)
        out = self.drop(out)

        out_1 = self.lin1(out)
        out_1 = self.relu1(out_1)
        out_1 = self.drop1(out_1)

        out_1 = out_1 + out

        out_2 = self.lin2(out_1)
        out_2 = self.relu2(out_2)
        out_2 = self.drop2(out_2)

        out_2 = out_2 + out_1

        out_f = self.li(out_2)

        return out_f

class MixModelDefault_Test(nn.Module):
    """
        Mixer model composed of spatial, temporal and fusion models
    """

    def __init__(self, num_classes, debug = False):
        super(MixModelDefault_Test, self).__init__()

        self.debug = debug

        self.spatial =  models.video.r3d_18(pretrained=True)
        self.temporal =  models.video.r3d_18(pretrained=True)
        self.fusion = Minimi(num_classes)

        self.relu =  nn.ReLU()
        self.soft1 = nn.Softmax()
        self.soft2 = nn.Softmax()
        self.batchN = nn.BatchNorm3d(3, affine=False)

        self.lin1 = nn.Linear(400,num_classes)
        self.lin2 = nn.Linear(400,num_classes)
        
        self.spa_s = torch.nn.Parameter(torch.ones(1))
        self.temp_s = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        x[1] = self.batchN(x[1])

        x_spa = self.spatial(x[0])
        x_temp =  self.temporal(x[1])

        x_spa = self.lin1(x_spa)
        x_temp = self.lin2(x_temp)

        x_spa = self.soft1(x_spa)
        x_temp = self.soft2(x_temp)

        x_res = x_spa + x_temp
        #x_res = torch.cat((x_spa, x_temp), 1)
        return x_res

        #return self.fusion(x_res)