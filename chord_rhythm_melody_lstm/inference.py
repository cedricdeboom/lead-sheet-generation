import numpy as np
import argparse
import yaml
import pickle
import os
import sys
import shutil

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import util

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import LSTM, MusicDataset, sort_batch, to_cuda
from mxl_generator import ChordRhythmMelodyMXL

###########
### BOOKKEEPING
###########

parser = argparse.ArgumentParser(description='LSTM inferences for melody model')
parser.add_argument('config', type=str,
                help='Required configuration YAML file')
parser.add_argument('model_file', type=str,
                help='Required model file')
parser.add_argument('seed_file', type=str,
                help='Required seed sequence (numpy)')
parser.add_argument('seed_index', type=int,
                help='Required seed index')
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
        input_dim_1=cfg['model']['input_dim_1'],
        input_dim_2=cfg['model']['input_dim_2'],
        hidden_dim=cfg['model']['lstm_dim'],
        batch_size=1,
        output_dim=cfg['model']['output_dim'],
        num_layers_bi=cfg['model']['num_layers_bi'],
        num_layers_lstm=cfg['model']['num_layers_lstm'],
        inference=True
    )

model.load_state_dict(torch.load(args.model_file))
model.eval()

print('-> READ DATA')
dataset = MusicDataset(args.seed_file, cfg['hyperparams']['sequence_length'], False)


### BOOTSTRAPPING
print('-> INFERENCE')

# get seed sequence and reset states
model.hidden_bi, model.hidden_lstm = model.init_hidden()
X_in = torch.FloatTensor(dataset.process_sequence(dataset.data[args.seed_index])[0]).unsqueeze(0)

# get bi-LSTM output (chord & rhythm processing)
bi_output = model.get_bi_output(X_in)


### SAMPLING LOOP

sampled_notes = []

# reset states
model.hidden_bi, model.hidden_lstm = model.init_hidden()

# begin with start token
lstm_out = model.process_lstm_sequence(
    bi_output[:, 0:1, :],
    torch.FloatTensor(dataset.start_token).unsqueeze(0),
    temperature=args.temperature,
    propagate_hidden=True
)

# sample melody note
print(torch.exp(lstm_out)[0, 0].shape)
next_char = torch.multinomial(torch.exp(lstm_out)[0, 0], 1)
sampled_notes.append(util.midi_pitch_to_pitch(int(next_char.detach().numpy())))
next_one_hot = torch.FloatTensor(np.zeros((1, 1, 130), dtype=np.float32))
next_one_hot[0, 0, next_char] = 1.0

for n in range(1, bi_output.shape[1]):
    lstm_out = model.process_lstm_sequence(bi_output[:, n:n+1, :], next_one_hot, temperature=args.temperature, propagate_hidden=True)
    next_char = torch.multinomial(torch.exp(lstm_out)[0, 0], 1)
    sampled_notes.append(util.midi_pitch_to_pitch(int(next_char.detach().numpy())))
    next_one_hot[:, :, :] = 0.0
    next_one_hot[0, 0, next_char] = 1.0


### CONVERT TO READABLE FORMAT
print('-> WRITE OUTPUT MXL')

# get original chord and rhythm sequence
seed_sequence = dataset.data[args.seed_index]

# gather rhythm and chords
chord_rhythm_sequence = []
for s in seed_sequence:
    rhythm = np.argmax(s[130:143])
    chord = np.argmax(s[143:])
    chord_rhythm_sequence.append((rhythm, chord))

# write output mxl
mxl_processor = ChordRhythmMelodyMXL(args.xml_template)
mxl_processor.process(chord_rhythm_sequence, sampled_notes)
mxl_processor.write_mxl(args.output_file)
