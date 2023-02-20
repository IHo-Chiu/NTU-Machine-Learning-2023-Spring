# -*- coding: utf-8 -*-

# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

# For feature selection
from sklearn.feature_selection import VarianceThreshold, GenericUnivariateSelect, r_regression, f_regression, mutual_info_regression

import xgboost as xgb

do_test = True
do_train = True

"""# Some Utility Functions

You do not need to modify this part.
"""

def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

"""# Dataset"""

class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

"""# Neural Network Model
Try out different model architectures by modifying the class below.
"""

class My_Model(nn.Module):
    def __init__(self, structure, dropout=0.0, do_batch_norm=False):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = []
        self.batchNorms = []
        for i in range(len(structure)-2):
            self.layers.append(nn.Linear(structure[i], structure[i+1]))
            self.batchNorms.append(nn.BatchNorm1d(structure[i+1]))
        self.layers.append(nn.Linear(structure[len(structure)-2], structure[len(structure)-1]))
        self.layers = nn.ModuleList(self.layers)
        self.batchNorms = nn.ModuleList(self.batchNorms)
        self.do_batch_norm = do_batch_norm
            
        self.actiFunc = nn.LeakyReLU()
        # self.actiFunc = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.do_batch_norm:
                x = self.batchNorms[i](x)
            x = self.actiFunc(x)
            x = self.dropout(x)
            
        x = self.layers[len(self.layers)-1](x)    
        x = x.squeeze(1) # (B, 1) -> (B)
        return x

"""# Feature Selection
Choose features you deem useful by modifying the function below.
"""

def select_feat(train_data, valid_data, test_data, select_all=True, method=None, k_best=1):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        sel = GenericUnivariateSelect(method, mode='k_best', param=k_best)
        sel.fit(train_data[:,37:-1], y_train)
        feat_idx = [int(x[1:])+37 for x in sel.get_feature_names_out()]
        
    
    print(feat_idx)
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid

"""# Training Loop"""

def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # Define your loss function, do not modify this.

    # Define your optimization algorithm. 
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    # optimizer = torch.optim.NAdam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    writer = SummaryWriter() # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        # writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            break
        
    print(f'best_loss = {best_loss}')
    return best_loss

"""# Configurations
`config` contains hyper-parameters for training and the path to save your model.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.1,   # validation_size = train_size * valid_ratio
    'n_epochs': 10000,     # Number of epochs.            
    'batch_size': 256,
    'dropout': 0.0,
    'learning_rate': 1e-3,
    'model_structure': [16, 8, 1],
    'do_batch_norm': False,
    'early_stop': 500,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}

"""# Dataloader
Read data from files and set up training, validation, and testing sets. You do not need to modify this part.
"""

same_seed(config['seed'])
train_data, test_data = pd.read_csv('./covid_train.csv').values, pd.read_csv('./covid_test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
# feature_selection_methods = [r_regression, f_regression, mutual_info_regression]
feature_selection_methods = [mutual_info_regression]
best_loss_list = []
for i, feature_selection_method in enumerate(feature_selection_methods):
    best_loss_list.append([])
    # for k_best in range(10,30):
    for k_best in range(27,28):
    
        x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'], feature_selection_method, k_best)

        # Print out the number of features.
        print(f'number of features: {x_train.shape[1]}')

        train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                                    COVID19Dataset(x_valid, y_valid), \
                                                    COVID19Dataset(x_test)

        # Pytorch data loader loads pytorch dataset into batches.
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

        if do_train:
            model = My_Model(structure=[x_train.shape[1]]+config['model_structure'], dropout=config['dropout'], do_batch_norm=config['do_batch_norm']).to(device) # put your model and data on the same computation device.
            best_loss = trainer(train_loader, valid_loader, model, config, device)
            best_loss_list[i].append(best_loss)
            
            print(best_loss_list)
            
        


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

if do_test:
    model = My_Model(structure=[x_train.shape[1]]+config['model_structure'], dropout=config['dropout'], do_batch_norm=config['do_batch_norm']).to(device)
    model.load_state_dict(torch.load(config['save_path']))
    preds = predict(test_loader, model, device) 
    save_pred(preds, 'pred.csv')  

