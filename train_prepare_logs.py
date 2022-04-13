import timeit
from datetime import datetime
import socket
import os
import glob
from network.SNN_model import SNN_ModelT

from network.SNN_Recurrent import SNNR_ModelT
from network.norse.CSNN_model import CSNN_ModelT
from network.norse.C3SNN_model import C3SNN_ModelT, C3DSNN_ModelT, C3DSNN_ModelT2, C3DSNN_C3D_ModelT, C3DSNN_Fire_ModelT
from network.own.C3NN_Base_model import C3DNN, C3DNN_NB, C3DNN_Small, C3DNN_NB_Small, C3DNN_Small_Alt, C3DNN_Med_Alt, ResNet_CNN

from network.own.CNN_LSTM_Base_model import CNN_LSTM, CNN_LSTM_Alt

from network.snntorch.C3SNN_SNN_model import C3SNN_SNNT_ModelT

from network.SP_C3_NN import SP_NN

from network.Mixed_C3D_model import Mixed_C3D

from network.CNN_Norse_model import ResNet_SNN


from network.C3D_model import C3D
from network.C3NN_model import C3NN_Mod
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset

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

nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
useVal = True

useWholeTimeSet = True

nTestInterval = 2 # Run on test set every nTestInterval epochs
snapshot = 5 # Store a model every snapshot epochs
lr = 2e-4 # Learning rate

dataset = 'hmdb51' # Options: hmdb51 or ucf101


#########################
#      N. Classes       #
#########################

dataset_classes = {'hmdb51' : 51, 'hmdb51_flow' : 51,  'kth' : 6, 'ucf101': 101}

if dataset in dataset_classes:
    num_classes = dataset_classes[dataset]
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError


save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'ResNet_CNN' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

def run_net(net, data, num_classes):
    """
        net (torch): SNN model
        data (torch.tensor): [batch_size, channels, timestep, height, width]
    """
    global device
    global useWholeTimeSet

    cops = torch.zeros(data.shape[0],data.shape[2], num_classes).to(device)
    data = torch.transpose(data, 1, 2)
    # Data shape is now [batch_size, timestep, channels, height, width]
    data /= 255

    
    if useWholeTimeSet:
        # Comment if LSTM
        data = torch.transpose(data, 1, 2)
        # Data shape is now [batch_size, channels, timestep, height, width]
        cops = net(data)
    else:
        for i in range(0, data.shape[1]):
            torch.cuda.empty_cache()
            #print(f"El minimo en los datos es {torch.min(data[:,i,:])}")
            output = net(data[:,i,:])
            #print(f"Output JEJE:\n {output}")
            cops[:,i,:] += output
        results = torch.as_tensor(cops)
        cops, _ = torch.max(results, 1)
        #cops = torch.mean(results, 1)
    #print(net.print_params())
    #print(f"Outputs:\n {cops[0]}\n")
    #print("###################")
    #input("SANTA")
    #_, preds = torch.max(cops, 1)
    #print(f"\nDevolviendo: \n{cops}")
    #q=""
    #while len(q) == 0:
    #    q = input("MMMM")
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
        print(f'Execution time:  {str(stop_time - start_time)}\n')
        return result
    return wrap

@execution_timer
def train_model(model, train_dataloader):

    train_size = len(train_dataloader.dataset)
    assert train_size > 0

    # reset the running loss and corrects
    running_loss = 0.0
    running_corrects = 0.0

    scheduler.step()
    model.train()

    for inputs, labels in tqdm(trainval_loaders[phase]):

        inputs = Variable(inputs, requires_grad=True).to(device)
        labels = Variable(labels).to(device)

        
        optimizer.zero_grad()

        cops = run_net(model, inputs, num_classes)

        preds = torch.max(cops, 1)[1]

        loss = criterion(cops.double(), labels.long())
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    

    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size
    
    # Log the data
    writer.add_scalar('data/'+phase+'_loss_epoch', epoch_loss, epoch)
    writer.add_scalar('data/'+phase+'_acc_epoch', epoch_acc, epoch)

    print("[train]  Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))





@execution_timer
def test_model(model, test_dataloader, is_val = False):
    
    test_size = len(test_dataloader.dataset)
    assert test_size > 0 and not is_val
    
    if is_val and test_size == 0:
        return [("val_loss", 0), ("val_acc", 0)]
    
    
    model.eval()

    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in tqdm(test_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        labels_c = labels.clone().detach()
        
        with torch.no_grad():
            outputs = run_net(model, inputs,num_classes)
            
        preds = torch.max(outputs, 1)[1]
        
        loss = criterion(outputs, labels_c.long())

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels_c)


    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects.double() / test_size

    # TODO Modify the following in order to log the outputs better 
    writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
    writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

    stage = "val" if is_val else "test"
    
    print("[{}] Loss: {} Acc: {}".format(stage,epoch_loss, epoch_acc))
    return [(stage+"_loss", epoch_loss), (stage+"_acc", epoch_acc)]
    

def save_model(epoch, model, optimizer, save_name, save_dir):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
    }, os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth.tar'))
    print("Save model at {}\n".format(os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth.tar')))



def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    global device

   
    model = ResNet_CNN(num_classes)
    train_params = [{'params': model.parameters(), 'lr': lr},]
    
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.Adam(train_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=16, preprocess = False), batch_size=16, shuffle=True, num_workers=4)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=16, num_workers=4)
    
    if useVal:
        val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=16), batch_size=16, num_workers=4)



   
        
    
    



    # Training loop
    for epoch in range(resume_epoch, num_epochs):
        print("Epoch: {}/{} ".format(epoch+1, nEpochs)")
        
        train_model(model, train_dataloader)

        if useVal:
            test_model(model, val_dataloader, is_val = True)

        if useTest and epoch % test_interval == (test_interval - 1):
            test_model(model, test_dataloader)

        
        if epoch % save_epoch == (save_epoch - 1):
            save_model(epoch, model, optimizer, saveName, save_dir)

        

    writer.close()


if __name__ == "__main__":
    train_model()