import numpy as np
import argparse
import yaml
import pickle
import os
import sys
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tensorboardX import SummaryWriter
from model import LSTM, MusicDataset, sort_batch, to_cuda

###########
### BOOKKEEPING
###########

parser = argparse.ArgumentParser(description='LSTM trainer for chord+rhythm+melody modeling')
parser.add_argument('config', type=str,
                help='Required configuration YAML file')
parser.add_argument('experiment_name', type=str,
                help='Required experiment name')
parser.add_argument('--overwrite', action="store_true",
                help='Overwrite existing experiment with the same name')

args = parser.parse_args()

if os.path.isdir(args.experiment_name) and not args.overwrite:
    print("Experiment already exists! Use --overwrite to clear existing experiment data.")
    sys.exit(0)
elif os.path.isdir(args.experiment_name):
    shutil.rmtree(args.experiment_name)
    if os.path.isdir(os.path.join('logs', args.experiment_name)):
        shutil.rmtree(os.path.join('logs', args.experiment_name))

os.mkdir(args.experiment_name)

log_writer = SummaryWriter(os.path.join('logs', args.experiment_name))

with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

shutil.copyfile(args.config, os.path.join(args.experiment_name, 'config.yml'))

np.random.seed(cfg['general']['random_seed'])
torch.manual_seed(cfg['general']['random_seed'])


###########
### START
###########

# Cross-entropy loss
# Takes -log- softmax logits as input !
def ce_loss(output, target, lenghts):
    logit_targets = output * target
    ce = torch.mean(torch.sum(-torch.sum(logit_targets, dim=2), dim=1) / lenghts)
    return ce

print('-> CREATE LSTM MODEL')
model = LSTM(
        input_dim_1=cfg['model']['input_dim_1'],
        input_dim_2=cfg['model']['input_dim_2'],
        hidden_dim=cfg['model']['lstm_dim'],
        batch_size=cfg['model']['batch_size'],
        output_dim=cfg['model']['output_dim'],
        num_layers_bi=cfg['model']['num_layers_bi'],
        num_layers_lstm=cfg['model']['num_layers_lstm'],
        inference=False
    ).cuda()

print('-> READ DATA')
dataset = MusicDataset(
        cfg['data']['processed_numpy_train_file'],
        cfg['hyperparams']['sequence_length'],
        cfg['data']['data_augmentation']
    )
dataloader = DataLoader(dataset, batch_size=cfg['model']['batch_size'], shuffle=False)

## validation set
val_dataset = MusicDataset(
        cfg['data']['processed_numpy_val_file'],
        cfg['hyperparams']['sequence_length'],
        cfg['data']['data_augmentation'],
        use_len_data=True
    )
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print('-> START TRAINING')
if cfg['hyperparams']['optimiser'] == 'adam':
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg['hyperparams']['learning_rate'])

model.train()
for batch_idx, batch_data in enumerate(dataloader):
    # zero grad model
    optimiser.zero_grad()
    
    # re-init hidden states
    model.hidden = model.init_hidden()
    
    # sort batch based on sequence length
    sort_batch(batch_data)

    # put batch on GPU
    batch_data = to_cuda(batch_data)

    # feed batch through model
    Y_output = model(batch_data[0], batch_data[2], cfg['hyperparams']['sequence_length'])
    Y_target = batch_data[1]
    Y_lenghts = batch_data[2]
    
    # calculate loss
    loss = ce_loss(Y_output, Y_target, Y_lenghts)
    
    # backprop
    loss.backward()
    optimiser.step()
    
    # log and validate
    if batch_idx % 100 == 0:
        model.eval()

        val_loss = []
        for val_batch_idx, val_batch_data in enumerate(val_dataloader):
            model.hidden = model.init_hidden()
            sort_batch(val_batch_data)
            val_batch_data = to_cuda(val_batch_data)
            Y_output = model(val_batch_data[0], val_batch_data[2], cfg['hyperparams']['sequence_length'])
            Y_target = val_batch_data[1]
            Y_lenghts = val_batch_data[2]
            val_loss.append(ce_loss(Y_output, Y_target, Y_lenghts).item())
        val_loss = np.asarray(val_loss).mean()
        
        print("Epoch ", batch_idx, "CE: ", loss.item(), "VAL_CE: ", val_loss)
        log_writer.add_scalar('train_loss', loss.item(), batch_idx)
        log_writer.add_scalar('val_loss', val_loss, batch_idx)

        model.train()

    # save model
    if batch_idx % 5000 == 0 and batch_idx >= 5000:
        torch.save(model.state_dict(), os.path.join(args.experiment_name, f'epoch_{batch_idx}.model'))
