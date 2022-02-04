import numpy as np 
import pandas as pd
from tqdm import tqdm

import cv2

import torch
import torch.nn as nn
from torch.utils import data
import copy

import torchvision
from torchvision import datasets, models, transforms
import accountant
import torch.multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')

import argparse
import gc

device = 'cuda'


torch.hub.set_dir('C:/Code/CS134/temp')

class ImagesDS(data.Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.len = len(files)
        self.transform = transform

    def __getitem__(self, index):
        
                
        return torch.from_numpy(np_image)

    def __len__(self):
        # The number of samples in the dataset.
        return self.len

class RandomCrop(object):
    def __init__(self, im_size):
        self.im_size = im_size
    def __call__(self, sample):
        c_max, r_max = sample[0].shape
        c, r = np.random.randint(0, c_max-self.im_size), np.random.randint(0, r_max-self.im_size)
        sample = sample[:,c:c+self.im_size,r:r+self.im_size]
        return sample

class Resize(object):
    def __init__(self, im_size):
        self.im_size = im_size
    def __call__(self, sample):
        return cv2.resize(sample, (self.im_size[0], self.im_size[1]))
    
class RandomFlip(object):
    def __init__(self, proba = 0.5):
        self.proba = proba
    def __call__(self, sample):
        # Get rid of one of these
        if(np.random.random() < self.proba):
            sample = np.flip(sample, 1)
        if(np.random.random() < self.proba):
            sample = np.flip(sample, 2)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        return (torch.from_numpy(sample.copy()).float())

def to_numpy(x):
    return x.cpu().detach().numpy()
def from_numpy(x):
    return torch.from_numpy(np.array(x)).float().to(device)
def score(x, y):
    return (np.mean(np.equal(np.argmax(x, axis = 1), y)))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def train_model(model, train_loader, val_loader, optimizer, num_epochs, batch_size, criterion, early_stop_rounds = 1, start_epoch = 0):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = np.inf
    epochs_since_best = 0
    dataloaders = {'train':train_loader, 'val':val_loader}
    dataset_sizes = {'train': int(len(dataloaders['train'].dataset)/batch_size + 1), 'val' : int(len(dataloaders['val'].dataset)/batch_size + 1)}
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()
            else:
                model.eval() 

            running_loss = 0
            bar = tqdm(enumerate(dataloaders[phase]), total=dataset_sizes[phase])
            for i, (x, y) in bar:
                #batch_size x n_channel x width x height
                x,y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(x)
                    np_outputs = to_numpy(outputs)
                    loss = criterion(outputs, y)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                running_loss += loss.item() #* x.size(0)
                if(i % 5 == 0):
                    bar.set_description(str(running_loss / (i+1)))
            epoch_loss = running_loss / dataset_sizes[phase]

            accountant.add_accuracy(epoch_loss, phase)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            # deep copy the model
            if phase == 'val' and epoch_loss < best_acc:
                print('Saving Checkpoint Model.')
                best_acc = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, 'checkpoints/'+accountant.name+ '_epoch'+str(start_epoch + epoch) +'.cp')
                epochs_since_best = 0
            elif phase == 'val':
                if(epochs_since_best >= early_stop_rounds):
                    model.load_state_dict(best_model_wts)
                    return model
                epochs_since_best = epochs_since_best + 1
                
        print()
    model.load_state_dict(best_model_wts)
    return model

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("accountant_file")
    parser.add_argument("-retrain_model", type = str, default = None)
    parser.add_argument("-start_epoch", type = int, default = 0)
    parser.add_argument("-tag", type = str, default = '')
    args = parser.parse_args()

    accountant.load_accountant(args.accountant_file, retrain_model = args.retrain_model, tag = args.tag)

    train_composed = None
    val_composed = None

    train_composed = transforms.Compose([Resize(accountant.image_size)])
    val_composed = transforms.Compose([Resize(accountant.image_size)])
    
    ds_train = ImagesDS(train_files, train_composed)
    ds_val = ImagesDS(train_files, val_composed)
    model = accountant.model
    print(model)

    model.to(device)


    batch_size = accountant.batch_size
    train_loader = data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = accountant.optimizer
    model = train_model(model, train_loader, val_loader, 
                         optimizer, accountant.train_epochs, batch_size, accountant.criterion)
    torch.save(model, 'models/'+accountant.name+'.pt')
    accountant.save_history('history/'+accountant.name+'.pkl')