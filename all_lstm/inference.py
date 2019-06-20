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
parser.add_argument('n_samples', type=int,
                help='Required number of samples to draw')
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
        num_layers_lstm=cfg['model']['lstm_layers'],
        inference=True
    )

model.load_state_dict(torch.load(args.model_file))
model.eval()

print('-> READ DATA')
dataset = MusicDataset(args.seed_file, cfg['hyperparams']['sequence_length'], False)


### BOOTSTRAPPING
print('-> INFERENCE')

# get seed sequence and reset states
model.hidden = model.init_hidden()
X_in = torch.FloatTensor(dataset.data[args.seed_index]).unsqueeze(0)

print(X_in.shape)


### SAMPLING LOOP

for n in range(args.n_samples):
    model.hidden = model.init_hidden()

    # feed sequence through RNN and get last output
    o = torch.exp(model.forward(X_in, None, None, temperature=args.temperature)[0, -1, :])

    # sample melody, rhythm and chord
    melody = torch.multinomial(o[:130], 1)[0]
    rhythm = torch.multinomial(o[130:143], 1)[0]
    chord = torch.multinomial(o[143:], 1)[0]

    if melody == 129 or chord == 48 or rhythm == 12:  #enforce consistent barlines
        melody = 129
        rhythm = 12
        chord = 48

    # generate one-hot vector
    one_hot = torch.zeros([1, 1, len(o)], dtype=torch.float32)
    one_hot[0, 0, melody] = 1.0
    one_hot[0, 0, 130 + rhythm] = 1.0
    one_hot[0, 0, 143 + chord] = 1.0

    X_in = torch.cat((X_in, one_hot), 1)


### CONVERT TO READABLE FORMAT
print('-> WRITE OUTPUT MXL')

final_seq = X_in.detach().numpy()

# gather rhythm and chords
chord_rhythm_sequence = []
note_sequence = []
for s in final_seq[0]:
    melody = np.argmax(s[:130])
    rhythm = np.argmax(s[130:143])
    chord = np.argmax(s[143:])
    chord_rhythm_sequence.append((rhythm, chord))
    note_sequence.append(util.midi_pitch_to_pitch(int(melody)))

# write output mxl
mxl_processor = ChordRhythmMelodyMXL(args.xml_template)
mxl_processor.process(chord_rhythm_sequence, note_sequence)
mxl_processor.write_mxl(args.output_file)


