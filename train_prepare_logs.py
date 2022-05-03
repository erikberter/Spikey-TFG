import timeit
from datetime import datetime
import socket
import os
import glob

from configparser import ConfigParser

from yaml import parse


from network.own.C3NN_Base_model import ResNet_CNN, C3DNN_Small, RPlus_CNN
from network.norse.C3SNN_model import C3SNN_ModelT, C3SNN_ModelT_scaled, C3SNN_ModelT_paramed, C3DSNN_Whole
from network.CNN_Norse_model import ResNet_SNN, ResNet_SNN_InverseScale, ResNet_Cop_SNN, SNN_Cont

from network.MixModels.mixer_models import MixModelDefault
from network.MixModels.mixer_models_SNN import MixModelDefaultSNN, MixModelAltSNN, MixModelDefaultSNNBig


from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from dataloaders.MixDataset import MixedDataset

import numpy as np

from functools import wraps

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
if device == 'cuda:0':
    torch.cuda.empty_cache()


torch.set_printoptions(precision=3, sci_mode=False)

#########################
#      Parameters       #
#########################

useTest = True # See evolution of the test set when training
useVal = True

useWholeTimeSet = True

#########################
#      N. Classes       #
#########################

dataset_classes = {'hmdb51' : 51,'hmdb51_2' : 51,'hmdb51_3' : 51, 'hmdb51_flow' : 51,  'kth' : 6, 'ucf101': 101, 'kth_rbg_diff' : 6, 
        'hmdb51_rbg_diff' : 51, 'kith_small':2, 'kith_rgb_small':2}



save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# Antes habia ID's, pero no no funcionaba y todas estaban en la 11.
# Voy a empezar a poner todas en la 42
save_dir = os.path.join(save_dir_root, 'run', 'run_43')

def run_net(net, data, num_classes):
    """
        net (torch): SNN model
        data (torch.tensor): [batch_size, channels, timestep, height, width]
    """
    global device
    global useWholeTimeSet

    if type(data) == list:
        # Is Mixed
        cops = torch.zeros(data[0].shape[0],data[0].shape[2], num_classes).to(device)
        data[0] /= 255
        data[1] /= 255
    else:
        cops = torch.zeros(data.shape[0],data.shape[2], num_classes).to(device)
    
        data /= 255

    if useWholeTimeSet:
        # Uncomment if LSTMif is_mixed:
        # Or better to put the transpose inside
        # data = torch.transpose(data, 1, 2)
        # Data shape is now [batch_size, channels, timestep, height, width]
        cops = net(data)
    else:
        for i in range(0, data.shape[1]):
            torch.cuda.empty_cache()
            
            output = net(data[:,i,:])
            
            cops[:,i,:] += output
        results = torch.as_tensor(cops)
        cops, _ = torch.max(results, 1)
        
    return cops

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y.cpu()]


def execution_timer(func):
    """
        Wrapper to print function time
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        print(f'Execution time:  {str(end_time - start_time)}\n')
        return result
    return wrap

@execution_timer
def train_model(epoch, model, num_classes, train_dataloader, scheduler, optimizer, criterion, writer, is_mixed = False):

    train_size = len(train_dataloader.dataset)
    assert train_size > 0

    # reset the running loss and corrects
    running_loss = 0.0
    running_corrects = 0.0

    scheduler.step()
    model.train()
    i = 1
    for inputs, labels in (sbar := tqdm(train_dataloader)):
        sbar.set_description("Curr Acc %s " % str(100*running_corrects/(i*6)))
        i += 1
        if is_mixed:
            inputs = [
                Variable(inputs[0], requires_grad=True).to(device),
                Variable(inputs[1], requires_grad=True).to(device)
                ]
        else:
            inputs = Variable(inputs, requires_grad=True).to(device)
        labels = Variable(labels).to(device)

        
        optimizer.zero_grad()

        cops = run_net(model, inputs, num_classes)

        preds = torch.max(cops, 1)[1]

        loss = criterion(cops.double(), labels.long())
        
        loss.backward()
        optimizer.step()

        if is_mixed:
            running_loss += loss.item() * inputs[0].size(0)
        else:
            running_loss += loss.item() * inputs.size(0)

        running_corrects += torch.sum(preds == labels)

    

    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size
    
    writer.add_scalar("Loss/train_loss", epoch_loss, epoch)
    writer.add_scalar("Acc/train_acc", epoch_acc, epoch)

    print("[train] Loss: {} Acc: {}".format(epoch_loss, epoch_acc))





@execution_timer
def test_model(epoch, model, num_classes, test_dataloader, criterion, writer,  is_val = False, is_mixed = False):

    test_size = len(test_dataloader.dataset)
    assert test_size > 0 and not is_val
    
    if is_val and test_size == 0:
        return [("val_loss", 0), ("val_acc", 0)]
    
    
    model.eval()

    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in tqdm(test_dataloader):
        if is_mixed:
            inputs = [
                Variable(inputs[0], requires_grad=True).to(device),
                Variable(inputs[1], requires_grad=True).to(device)
                ]
        else:
            inputs = Variable(inputs, requires_grad=True).to(device)
        labels = Variable(labels).to(device)
        
        labels_c = labels.clone().detach()
        
        with torch.no_grad():
            outputs = run_net(model, inputs,num_classes)
            
        preds = torch.max(outputs, 1)[1]
        
        loss = criterion(outputs, labels_c.long())

        
        if is_mixed:
            running_loss += loss.item() * inputs[0].size(0)
        else:
            running_loss += loss.item() * inputs.size(0)

        running_corrects += torch.sum(preds == labels_c)


    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects.double() / test_size

    stage = "val" if is_val else "test"

    writer.add_scalar("Loss/" + stage+"_loss", epoch_loss, epoch)
    writer.add_scalar("Acc/" + stage+"_acc", epoch_acc, epoch)

    
    
    print("[{}] Loss: {} Acc: {}".format(stage,epoch_loss, epoch_acc))
    

def save_model(epoch, model, optimizer, save_name, save_dir):
    global config

    save_path = os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth.tar')
    
    config['session']['last_epoch'] = str(epoch + 1)
    save_config()

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
    }, save_path)
    
    print("Save model at {}\n".format(save_path))


def load_model(model, saveName, optimizer, resume_epoch):
    load_path = os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')
    
    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    
    print("Initializing weights from: {}...".format(load_path))
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_dict'])
    
    return model, optimizer

def load_dataloader(name, split, params, shuffle = False, is_mixed = False):
    
    if is_mixed:
        return DataLoader(
            MixedDataset(
                dataset=name, split=split,clip_len=params['clip_len']
                ), 
            batch_size=params['batch_size'], shuffle=shuffle, num_workers=4
            )
    
    return DataLoader(
        VideoDataset(
            dataset=name, split=split,clip_len=params['clip_len']
            ), 
        batch_size=params['batch_size'], shuffle=shuffle, num_workers=4
        )


def load_dataloaders(name, params, useVal = False, is_mixed = False):
    train_dataloader = load_dataloader(name, 'train', params, shuffle=True, is_mixed=is_mixed)
    test_dataloader = load_dataloader(name, 'test', params, is_mixed=is_mixed)

    val_dataloader = load_dataloader(name, 'val', params, is_mixed=is_mixed) if useVal else None

    return train_dataloader, test_dataloader, None


def train_session_model(
        modelName,
        model_class, 
        dataset='kth', 
        lr=2e-4,
        num_epochs=40, 
        save_epoch= 5, 
        useTest= True, 
        test_interval= 2,
        resume_epoch = 0,
        is_mixed = False,
        dataloader_params = {'clip_len' : 16, 'batch_size' : 16}
    ):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    global device
    global save_dir
    global config

    if dataset in dataset_classes:
        num_classes = dataset_classes[dataset]
    else:
        print(f"Not detected database {dataset}")
        print("Skipping...")
        return

    try:
        model = model_class(num_classes)
    except Exception as exp:
        print(exp)
        print(f"Error loading model {model_class.__name__}")
        print("Skipping...")
        return

    saveName = modelName + '-' + dataset
    log_folder =  model_class.__name__ + "_" + dataset + "_"+ str(lr) + "_"+ str(dataloader_params["clip_len"]) + "_" + str(dataloader_params["batch_size"])
    log_dir = os.path.join(save_dir, 'models',log_folder)
    writer = SummaryWriter(log_dir=log_dir)

    

    model.to(device)

    train_params = [{'params': model.parameters(), 'lr': lr},]
    

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.Adam(train_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        model, optimizer = load_model(model, saveName, optimizer, resume_epoch)
        

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # TODO Add hyperparameters to summary
    model.to(device)
    criterion.to(device)


    train_dataloader, test_dataloader, val_dataloader = load_dataloaders(dataset, dataloader_params, useVal = True, is_mixed = is_mixed)


    print('Training model on {} dataset...'.format(dataset))
    
    
    # Training loop
    for epoch in range(resume_epoch, num_epochs):
        print("Epoch: {}/{} ".format(epoch+1, num_epochs))
        
        if epoch == 6:
            if hasattr(model, 'activate_grad') and callable(getattr(model, 'activate_grad')):
                print('Activating gradients')
                model.activate_grad(True)

        # Training
        train_model(epoch, model, num_classes, train_dataloader, scheduler, optimizer, criterion, writer, is_mixed=is_mixed)
        
        #Validation
        if useVal and val_dataloader is not None:
            test_model(epoch, model, num_classes, val_dataloader,criterion, writer, is_val = True, is_mixed=is_mixed)

        # Test
        if useTest and epoch % test_interval == (test_interval - 1):
            test_model(epoch,model, num_classes, test_dataloader, criterion, writer, is_mixed=is_mixed)

        # Save model
        if epoch % save_epoch == (save_epoch - 1):
            save_model(epoch, model, optimizer, saveName, save_dir)

        

    writer.close()


# List of running sessions
# Elements of the list must be on the shape of
#   (Name, model, dataset, model_params, dataset_params)
#   (Name, model, dataset, 'lr', epochs, save_epoch,  useTest, test_interval,  resume_epoch,{clip_len, batch_size})
    
train_models = [
    (("Small_CNN", C3DNN_Small, "kth", 2e-4, 20, 5,  True, 2), {}),
    (("Small_CNN", C3DNN_Small, "hmdb51", 2e-4, 20, 5,  True, 2), {}),
    (("Small_CNN", C3DNN_Small, "ucf101", 2e-4, 20, 5,  True, 2), {}),
    (("ResNet_CNN", ResNet_CNN, "kth", 2e-4, 20, 5,  True, 2), {}),
    (("ResNet_CNN", ResNet_CNN, "hmdb51", 2e-4, 20, 2,  True, 2), {}),
    (("ResNet_CNN", ResNet_CNN, "ucf101", 2e-4, 20, 2,  True, 2), {}),
    (("C3SNN_Normal", C3SNN_ModelT, "kth", 2e-4, 20, 5, True, 2), {}),
    (("C3SNN_Scaled", C3SNN_ModelT_scaled, "kth", 2e-4, 20, 5, True, 2), {}),
    (("C3SNN_Normal", C3SNN_ModelT, "hmdb51", 2e-4, 20, 5, True, 2), {}),
    (("C3SNN_Scaled", C3SNN_ModelT_scaled, "hmdb51", 2e-4, 20, 5, True, 2), {}),
    (("C3SNN_Normal", C3SNN_ModelT, "ucf101", 2e-4, 20, 5, True, 2), {}),
    (("C3SNN_Scaled", C3SNN_ModelT_scaled, "ucf101", 2e-4, 20, 5, True, 2), {}),
    (("ResNet_SNN", ResNet_SNN, "kth", 2e-4, 20, 5, True, 2), {}),
    (("ResNet_SNN", ResNet_SNN, "hmdb51", 2e-4, 20, 3, True, 2), {}),
    (("ResNet_SNN", ResNet_SNN, "ucf101", 2e-4, 20, 2, True, 2), {}),
    (("Small_CNN", C3DNN_Small, "kth_rbg_diff", 2e-4, 20, 5,  True, 2), {}),
    (("C3SNN_Normal", C3SNN_ModelT, "kth_rbg_diff", 2e-4, 20, 5, True, 2), {}),
    (("C3SNN_Scaled", C3SNN_ModelT_scaled, "kth_rbg_diff", 2e-4, 20, 5, True, 2), {}),
    (("C3SNN_Paramed", C3SNN_ModelT_paramed, "kth_rbg_diff", 2e-4, 20, 5, True, 2), {}),
    (("ResNet_2CNN", MixModelDefault, "hmdb51", 2e-4, 20, 2,  True, 2), {'is_mixed' : True, 'dataloader_params' : {'clip_len' : 16, 'batch_size' : 6}}),
    (("ResNet_SNN_Inverse", ResNet_SNN_InverseScale, "hmdb51", 2e-4, 20, 3, True, 2), {}), # Not fully trained. Not worth it
    (("ResNet_2CNN", MixModelDefault, "ucf101", 2e-4, 20, 2,  True, 2), {'is_mixed' : True, 'dataloader_params' : {'clip_len' : 16, 'batch_size' : 6}}),
    (("C3DSNN_Whole", C3DSNN_Whole, "kith_small", 2e-4, 20, 5, True, 2), {'is_mixed' : False, 'dataloader_params' : {'clip_len' : 16, 'batch_size' : 6}}),
    (("C3DSNN_Whole", C3DSNN_Whole, "kith_rgb_small", 2e-4, 20, 5, True, 2), {'is_mixed' : False, 'dataloader_params' : {'clip_len' : 16, 'batch_size' : 6}}),
    (("ResNet_2CNN_SNN", MixModelDefaultSNN, "hmdb51", 2e-4, 20, 2,  True, 2), {'is_mixed' : True, 'dataloader_params' : {'clip_len' : 16, 'batch_size' : 6}}),
    (("ResNet_2CNN_SNN", MixModelDefaultSNN, "ucf101", 2e-4, 20, 2,  True, 2), {'is_mixed' : True, 'dataloader_params' : {'clip_len' : 16, 'batch_size' : 6}}),
    # Fracaso, el pretraining en las resnet 3d solo empeora todo
    (("ResNet_2CNN_SNN_Alt_Bad", MixModelAltSNN, "hmdb51", 2e-4, 20, 2,  True, 2), {'is_mixed' : True, 'dataloader_params' : {'clip_len' : 16, 'batch_size' : 6}}),
    (("ResNet_2CNN_SNN_Alt", MixModelDefaultSNNBig, "hmdb51", 2e-4, 20, 2,  True, 2), {'is_mixed' : True, 'dataloader_params' : {'clip_len' : 16, 'batch_size' : 6}}),
    (("ResNet_2CNN_SNN_Alt", MixModelDefaultSNNBig, "ucf101", 2e-4, 20, 2,  True, 2), {'is_mixed' : True, 'dataloader_params' : {'clip_len' : 16, 'batch_size' : 6}}),
    (("ResNet_SNN", ResNet_SNN, "hmdb51_2", 2e-4, 20, 4, True, 2), {}),
    (("ResNet_CNN", ResNet_CNN, "hmdb51_2", 2e-4, 20, 4,  True, 2), {}),
    (("ResNet_SNN", ResNet_SNN, "hmdb51_3", 2e-4, 20, 4, True, 2), {}),
    (("ResNet_CNN", ResNet_CNN, "hmdb51_3", 2e-4, 20, 4,  True, 2), {}),
    (("ResNet_SNN_Cop", ResNet_Cop_SNN, "hmdb51", 2e-4, 20, 4,  True, 2), {}),
    (("ResNet_SNN_Cop", ResNet_Cop_SNN, "hmdb51_2", 2e-4, 20, 4,  True, 2), {}),
    (("ResNet_SNN_Cop", ResNet_Cop_SNN, "hmdb51_3", 2e-4, 20, 4,  True, 2), {}),
    (("SNN_Cont", SNN_Cont, "kth_rbg_diff", 2e-4, 20, 4,  True, 2), {}),

    

    
   ]


def prepare_config():
    config = ConfigParser()
    
    if not os.path.exists('auto_session.ini'):
        config.add_section('session')
        config.set('session', 'last_model', '0')
        config.set('session', 'last_epoch', '0')
        save_config_pre(config)
    else:
        config.read('auto_session.ini')
    
    return config

def save_config():
    global config
    with open('auto_session.ini', 'w') as configfile:
        config.write(configfile)

def save_config_pre(config = None):
    with open('auto_session.ini', 'w') as configfile:
        config.write(configfile)

if __name__ == "__main__":
    
    config = prepare_config()
    
    for i, model_params in enumerate(train_models):
        print(f" Act {i} _ {model_params[0][0]} _ {model_params[0][2]}")
        if int(config['session']['last_model']) > i:
            continue
        if int(config['session']['last_model']) == i:
            print("Loading from resumed epoch")
            model_params[1]['resume_epoch'] = int(config['session']['last_epoch'])

        train_session_model(*model_params[0], **model_params[1])

        config['session']['last_model'] = str(int(config['session']['last_model']) + 1)
        config['session']['last_epoch'] = '0'
        
        save_config()

