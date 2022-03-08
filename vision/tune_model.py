from glob import glob
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
import torchvision.transforms as T
import accountant
import torch.multiprocessing as mp
from piece_data import ImagesListDS

import warnings
warnings.filterwarnings('ignore')

import argparse
import gc

device = 'cuda'


torch.hub.set_dir('C:/Code/CS134/temp')



class RandomCrop(object):
    def __init__(self, im_size):
        self.im_size = im_size
    def __call__(self, sample):
        c_max, r_max, _ = sample.shape
        print(sample.shape)
        c, r = np.random.randint(0, c_max-self.im_size), np.random.randint(0, r_max-self.im_size)
        sample = sample[c:c+self.im_size,r:r+self.im_size, :]
        return sample

class Resize(object):
    def __init__(self, im_size):
        self.im_size = im_size
    def __call__(self, sample):
        return cv2.resize(sample, (self.im_size, self.im_size))
    
class RandomFlip(object):
    def __init__(self, proba = 0.2):
        self.proba = proba
    def __call__(self, sample):
        # Get rid of one of these
        if(np.random.random() < self.proba):
            sample = np.flip(sample, 1)
        if(np.random.random() < self.proba):
            sample = np.flip(sample, 2)
        return sample

class RandomRotate(object):
    def __init__(self, proba = 1) -> None:
        self.proba = proba
    def __call__(self, sample):
        if(np.random.random() < self.proba):
            sample = np.rot90(sample, k = np.random.choice([0,1,2,3]))
        return sample
        

class ToTensor(object):
    def __call__(self, sample):
        t = torch.from_numpy(sample.copy()).permute(2,0,1)
        return t 

def to_numpy(x):
    return x.cpu().detach().numpy()
def from_numpy(x):
    return torch.from_numpy(np.array(x)).float().to(device)
def score(x, y):
    return (np.mean(np.equal(np.argmax(x, axis = 1), y)))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def train_model(model, train_loader, val_loader, optimizer, num_epochs, batch_size, criterion, early_stop_rounds = 2, start_epoch = 0):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = np.inf
    epochs_since_best = 0
    dataloaders = {'train':train_loader, 'val':val_loader}
    dataset_sizes = {'train': int(len(dataloaders['train'].dataset)/batch_size + 1), 'val' : int(len(dataloaders['val'].dataset)/batch_size + 1)}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
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
            for i, (anchor, positive, negative) in bar:
                #batch_size x n_channel x width x height
                anchor, positive, negative = normalize(anchor.float()/255.0).to(device), normalize(positive.float()/255.0).to(device), normalize(negative.float()/255.0).to(device)
                # anchor, positive, negative = normalize(anchor.permute(0, 3, 1, 2)).to(device), normalize(positive.permute(0, 3, 1, 2)).to(device), normalize(negative.permute(0, 3, 1, 2)).to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_anchor = model(anchor)
                    outputs_positive = model(positive)
                    outputs_negative = model(negative)
                    loss = criterion(outputs_anchor, outputs_positive, outputs_negative)
                    
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

    train_composed = transforms.Compose([Resize(accountant.image_size), RandomRotate(), ToTensor(), T.RandAugment(num_ops = 6)])
    val_composed = transforms.Compose([Resize(accountant.image_size), RandomRotate(), ToTensor()])
    
    ds_train = ImagesListDS(train_composed, multiplier=2.5)
    ds_val = ImagesListDS(val_composed, multiplier=1)
    model = accountant.model
    print(model)

    model.to(device)


    batch_size = accountant.batch_size
    train_loader = data.DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=5)
    val_loader = data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=5)
    optimizer = accountant.optimizer
    model = train_model(model, train_loader, val_loader, 
                         optimizer, accountant.train_epochs, batch_size, accountant.criterion)
    torch.save(model, 'models/'+accountant.name+'.pt')
    accountant.save_history('history/'+accountant.name+'.pkl')