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

from model import LSTM, MusicDataset, sort_batch, to_cuda
from mxl_generator import ChordRhythmMXL

###########
### BOOKKEEPING
###########

parser = argparse.ArgumentParser(description='LSTM trainer for chord+rhythm modeling')
parser.add_argument('config', type=str,
                help='Required configuration YAML file')
parser.add_argument('model_file', type=str,
                help='Required model file')
parser.add_argument('seed_file', type=str,
                help='Required seed sequence (numpy)')
parser.add_argument('seed_index', type=int,
                help='Required seed index')
parser.add_argument('n_samples', type=int,
                help='Required number of samples to draw')
parser.add_argument('temperature', type=float,
                help='Required sampling temperature')
parser.add_argument('random_seed', type=int,
                help='Required random seed')
parser.add_argument('xml_template', type=str,
                help='Required XML template file')
parser.add_argument('output_file', type=str,
                help='Required MXL output file name')

args = parser.parse_args()

if not os.path.exists(args.model_file):
    print("Model does not exist.")
    sys.exit(0)

with open(args.config, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)


###########
### START
###########

print('-> CREATE LSTM MODEL')
model = LSTM(
        input_dim=cfg['model']['data_dim'],
        hidden_dim=cfg['model']['lstm_dim'],
        batch_size=cfg['model']['batch_size'],
        output_dim=cfg['model']['data_dim'],
        num_layers=cfg['model']['lstm_layers'],
        inference=True
    )

model.load_state_dict(torch.load(args.model_file))
model.eval()

print('-> READ DATA')
with open(args.seed_file, 'rb') as f:
    id_to_sheet = pickle.load(f)
    data = pickle.load(f)


### BOOTSTRAPPING

# get seed sequence
numpy_seed_sequence = data[args.seed_index][:, 130:]
# convert to tensor + add batch dimension
seed_sequence = torch.FloatTensor(numpy_seed_sequence).unsqueeze(0)

print('-> INFERENCE')
### SAMPLING LOOP
for n in range(args.n_samples):
    # reset RNN hidden states
    model.hidden = model.init_hidden()

    # feed sequence through RNN and get last output
    o = torch.exp(model.forward(seed_sequence, None, None, temperature=args.temperature)[0, -1, :])

    # sample rhythm and chord
    rhythm = torch.multinomial(o[:13], 1)[0]
    chord = torch.multinomial(o[13:], 1)[0]

    if chord == 48 or rhythm == 12:  #enforce consistent barlines
        rhythm = 12
        chord = 48

    # generate one-hot vector
    one_hot = torch.zeros([1, 1, len(o)], dtype=torch.float32)
    one_hot[0, 0, rhythm] = 1.0
    one_hot[0, 0, 13 + chord] = 1.0

    seed_sequence = torch.cat((seed_sequence, one_hot), 1)

### CONVERT TO READABLE FORMAT
print('-> WRITE OUTPUT MXL')
# remove batch dimension and convert to numpy
seed_sequence = seed_sequence[0].detach().numpy()

# gather rhythm and chords
idx_sequence = []
for s in seed_sequence:
    rhythm = np.argmax(s[:13])
    chord = np.argmax(s[13:])
    idx_sequence.append((rhythm, chord))
    print(rhythm, chord)

# write output mxl
mxl_processor = ChordRhythmMXL(args.xml_template)
mxl_processor.process(idx_sequence)
mxl_processor.write_mxl(args.output_file)
