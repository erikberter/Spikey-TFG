import os

import pandas as pd
import numpy as np
import seaborn as sn

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from dataloaders.dataset import VideoDataset

from sklearn.metrics import confusion_matrix

from network.own.C3NN_Base_model import ResNet_CNN, C3DNN_Small, RPlus_CNN
from network.CNN_Norse_model import ResNet_SNN

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def get_dataloader(dataset, split):
    return DataLoader(
        VideoDataset( dataset=dataset, split=split,clip_len=16 ), 
        batch_size=16, shuffle=False, num_workers=4
    )


# TOOD Abstraer a funcion fuera 
def load_model(model, saveName, resume_epoch):
    load_path = os.path.join('run','run_43', 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')
    
    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    
    print("Initializing weights from: {}...".format(load_path))
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


dataset_classes = {'hmdb51' : 51,'hmdb51_2' : 51,'hmdb51_3' : 51, 'hmdb51_flow' : 51,  'kth' : 6, 'ucf101': 101, 'kth_rbg_diff' : 6, 
        'hmdb51_rbg_diff' : 51, 'kith_small':2, 'kith_rgb_small':2}



models = [
    (ResNet_CNN, 'hmdb51'), # Hecho
    (ResNet_CNN, 'hmdb51_3'),
    (ResNet_CNN, 'hmdb51_2'),
    (ResNet_SNN, 'hmdb51'), # Hecho
    (ResNet_SNN, 'hmdb51_3'),
    (ResNet_SNN, 'hmdb51_2'),
]


if __name__ == "__main__":

    for model_params in models:
        print(f"Estamos en el modelo {model_params[0].__name__}")
        dataset = model_params[1]

        model = model_params[0](dataset_classes[dataset])
        model.to(device)
        model = load_model(model, model_params[0].__name__ + '-' + dataset, 16)
        model.to(device)
        model.eval()

        for split in ['test', 'train']:

            y_true, y_pred = [], []

            print(f"Estamos en el split {split}")
            dataloader = get_dataloader(dataset, split)
            
            data_size = len(dataloader.dataset)

            running_corrects = 0
            test_size = len(dataloader.dataset)

            for inputs, labels in tqdm(dataloader):
                out = model(inputs.to(device)/255)

                output = (torch.max(out, 1)[1]).cpu().numpy()
                y_pred.extend(output)

                labels = labels.cpu().numpy()
                y_true.extend(labels)

            acc = np.sum(np.array(y_pred) == np.array(y_true))
            print(f"Hemos tenido un acc de {acc/data_size}")

            classes = []
            
            with open('dataloaders/labels/' + dataset + "_labels.txt", 'r') as f:
                for line in f:
                    classes += [line.split()[1]]

            cf_matrix = confusion_matrix(y_true, y_pred)
            df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                                columns = [i for i in classes])
            plt.figure(figsize = (12,7))
            sn.heatmap(df_cm, annot=False)
            print("Guardando el archivo")
            plt.savefig('data/confusion_matrix/' + model_params[0].__name__ + "_" + dataset + "_" + split + ".png")