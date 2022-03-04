import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils import data

import torchvision
from torchvision import datasets, models, transforms


image_size = None
train_epochs = 10
batch_size = 32
parent_model = None
name = None
model = None
optimizer = None
criterion = None
accuracy_history_train = []
accuracy_history_val = []
extra_history = []
eval = False
retrain = None

def print_accountant():
    print('Image Size', image_size)
    print('Batch Size', batch_size)
    print('Parent Mode', parent_model)
    print('Name', name)
    print('Evaluate', eval)
def set_layer_freeze(layer, freeze):
    params = list(layer.parameters())
    for param in params:
        param.requires_grad = not freeze

def set_freeze_exact(model, level):
    print('Exact thaw level : ', level)
    layers = list(model.children())
    set_layer_freeze(layers[-level], False)
    set_layer_freeze(layers[0], False)
    set_layer_freeze(layers[-1], False)

def set_freeze_above(model, level):
    print('Above thaw level : ', level)
    layers = list(model.children())
    for i, layer in enumerate(layers):
        if(i >= len(layers) - level):
            print(i, 'Unfrozen', layer)
            set_layer_freeze(layer, False)
        else:
            print(i, 'Frozen', layer)
            set_layer_freeze(layer, True)
    set_layer_freeze(layers[0], False)
    set_layer_freeze(layers[-1], False)

def make_trainable(model):
    for param in model.parameters():
        param.requires_grad = True

def retrain_model(parent_model, epoch):
    base_model = torch.load(parent_model)
    make_trainable(base_model)
    return base_model

def add_accuracy(accuracy, state):
    if(state == 'val'):
        accuracy_history_val.append(accuracy)
    else:
        accuracy_history_train.append(accuracy)
def add_extra(extra):
    extra_history.append(extra)
    
def load_accountant(file, evaluate = False, evaluate_celltype = -1, retrain_model = None, tag = ''):
    global eval
    global retrain
    global accuracy_history_train
    global accuracy_history_val
    global extra_history
    global name
    if(evaluate_celltype != -1):
        eval_celltype = int(evaluate_celltype)
        print('eval_CT', eval_celltype)
    if(evaluate):
        eval = True
    if(retrain_model is not None):
        retrain = retrain_model
    f = open(file, "r").read()
    exec(f, globals())
    if(retrain_model is not None):
        e = load_history('history/'+name+'.pkl')
        accuracy_history_train = e['train']
        accuracy_history_val = e['val']
        extra_history = e['extra']
    name = name + tag
    print_accountant()

def load_history(file):
    e = None
    with open(file, "rb") as input_file:
        e = pickle.load(input_file)
    return e

def save_history(file):
    with open(file, 'wb') as fp:
        pickle.dump({'train':accuracy_history_train, 'val': accuracy_history_val, 'extra':extra_history}, fp)