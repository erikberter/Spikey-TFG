from network.own.C3NN_Base_model import ResNet_CNN, C3DNN_Small, RPlus_CNN
from network.norse.C3SNN_model import C3SNN_ModelT, C3SNN_ModelT_scaled, C3SNN_ModelT_paramed, C3DSNN_Whole
from network.CNN_Norse_model import ResNet_SNN

from network.MixModels.mixer_models import MixModelDefault
from network.MixModels.mixer_models_SNN import MixModelDefaultSNN, MixModelAltSNN, MixModelDefaultSNNBig





models  = (
    (C3DNN_Small, 51),
    (C3SNN_ModelT_scaled, 51),
    (ResNet_SNN, 51),
    (ResNet_CNN,51),
    (MixModelDefault, 51),
    (MixModelDefaultSNNBig, 51),
)


for model_params in models:
    model = model_params[0](model_params[1])
    print(f"{model_params.__name__} & {sum(p.numel() for p in model.parameters()}")