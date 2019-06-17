import numpy as np
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Pytorch RNN axes:
# 1. The first axis is the sequence itself,
# 2. the second indexes instances in the mini-batch,
# 3. and the third indexes elements of the input.
# But we will put the batch axis in front (through the batch_first=True argument)

class LSTM(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dim, batch_size, output_dim, num_layers_bi=2, num_layers_lstm=2, inference=False):
        super(LSTM, self).__init__()
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers_lstm = num_layers_lstm
        self.num_layers_bi = num_layers_bi
        self.inference = inference
        
        self.hidden_bi, self.hidden_lstm = self.init_hidden()
        
        # Bidirectional LSTM
        if self.num_layers_bi > 0:
            self.lstm_bi = nn.LSTM(
                self.input_dim_1,
                self.hidden_dim,
                self.num_layers_bi,
                batch_first=True,
                bidirectional=True
            )
        else:
            self.lstm_bi = nn.LSTM(
                self.input_dim_1,
                self.hidden_dim,
                self.num_layers_lstm,
                batch_first=True,
                bidirectional=False
            )
        
        # Second pair of LSTM layers
        if self.num_layers_bi > 0:
            self.lstm = nn.LSTM(self.hidden_dim * 2 + self.input_dim_2, self.hidden_dim, self.num_layers_lstm, batch_first=True)
        else:
            self.lstm = nn.LSTM(self.hidden_dim + self.input_dim_2, self.hidden_dim, self.num_layers_lstm, batch_first=True)
        
        # Last dense layer
        self.dense = nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        if self.num_layers_bi > 0:
            return (
                (torch.zeros(self.num_layers_bi * 2, self.batch_size, self.hidden_dim),
                    torch.zeros(self.num_layers_bi * 2, self.batch_size, self.hidden_dim)),
                (torch.zeros(self.num_layers_lstm, self.batch_size, self.hidden_dim),
                    torch.zeros(self.num_layers_lstm, self.batch_size, self.hidden_dim))
                )
        else:
            return (
                (torch.zeros(self.num_layers_lstm, self.batch_size, self.hidden_dim),
                    torch.zeros(self.num_layers_lstm, self.batch_size, self.hidden_dim)),
                (torch.zeros(self.num_layers_lstm, self.batch_size, self.hidden_dim),
                    torch.zeros(self.num_layers_lstm, self.batch_size, self.hidden_dim))
                )

    def get_bi_output(self, input_X):
        # get chord and rhythm info from input
        input_X_1 = input_X[:, :, self.input_dim_2:]
        out_bi, _ = self.lstm_bi(input_X_1)
        
        return out_bi
    
    def process_lstm_sequence(self, bi_part, melody_part, temperature=1.0, propagate_hidden=False):
        concat_X_2 = torch.cat((melody_part, bi_part), 2)
            
        if propagate_hidden:
            out_lstm, self.hidden_lstm = self.lstm(concat_X_2, self.hidden_lstm)
        else:
            out_lstm, self.hidden_lstm = self.lstm(concat_X_2)
        
        # Generate sequence predictions
        X_pred = self.dense(out_lstm)
        
        # Apply nonlinearities
        X_softmax = F.log_softmax(X_pred / temperature, dim=2)
        
        return X_softmax

    def forward(self, input_X, lengths_X, max_length, temperature=1.0, propagate_hidden=False):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).
        
        # get chord and rhythm info from input
        input_X_1 = input_X[:, :, self.input_dim_2:]
        
        if not self.inference:
            packed_X_1 = torch.nn.utils.rnn.pack_padded_sequence(input_X_1, lengths_X, batch_first=True)
        else:
            packed_X_1 = input_X_1
            
        if propagate_hidden:
            out_bi, self.hidden_bi = self.lstm_bi(packed_X_1, self.hidden_bi)
        else:
            out_bi, self.hidden_bi = self.lstm_bi(packed_X_1)
        
        if not self.inference:
            unpacked_X_1, _ = torch.nn.utils.rnn.pad_packed_sequence(out_bi, batch_first=True, total_length=max_length)
        else:
            unpacked_X_1 = out_bi
        
        # Concat output of bi-LSTM with melody part of input
        input_X_2 = input_X[:, :, :self.input_dim_2]
        concat_X_2 = torch.cat((input_X_2, unpacked_X_1), 2)
        
        # Feed through second pair of LSTM layers
        if not self.inference:
            packed_X_2 = torch.nn.utils.rnn.pack_padded_sequence(concat_X_2, lengths_X, batch_first=True)
        else:
            packed_X_2 = concat_X_2
            
        if propagate_hidden:
            out_lstm, self.hidden_lstm = self.lstm(packed_X_2, self.hidden_lstm)
        else:
            out_lstm, self.hidden_lstm = self.lstm(packed_X_2)
        
        if not self.inference:
            unpacked_X_2, _ = torch.nn.utils.rnn.pad_packed_sequence(out_lstm, batch_first=True, total_length=max_length)
        else:
            unpacked_X_2 = out_lstm
        
        # Generate sequence predictions
        X_pred = self.dense(unpacked_X_2)
        
        # Apply nonlinearities
        X_softmax = F.log_softmax(X_pred / temperature, dim=2)
        
        return X_softmax


# NUMPY INPUT DATA SCHEMA
# [0:127] -> MIDI pitch (one-hot) - melody
# [128] -> rest (one-hot) - melody
# [129] -> barline (one-hot) - melody
# [130:141] -> rhythm index (one-hot) - rhythm
# [142] -> barline (one-hot) (repeated) - rhythm
# [143:190] -> chord (one-hot) - chord
# [191] -> barline (one-hot) (repeated) - chord

class MusicDataset(Dataset):
    def __init__(self, data_file, sequence_length, data_augmentation):
        super(MusicDataset, self).__init__()
        self.sequence_length = sequence_length
        self.data_augmentation = data_augmentation
        
        with open(data_file, 'rb') as f:
            self.id_to_sheet = pickle.load(f)
            self.data = pickle.load(f)
        
        self.data = [x.astype(np.float32) for x in self.data]
        self.start_token = np.zeros((1, 130), dtype=np.float32)
        self.start_token[:, 0] = 1.0
        
        # pad all sequences to desired sequence length
        self.mask_lengths = []
        for i, x in enumerate(self.data):
            if len(x) < self.sequence_length:
                s = x.shape
                self.data[i] = np.zeros((self.sequence_length, s[1]), dtype=np.float32)
                self.data[i][:s[0], :] = x
                self.mask_lengths.append(s[0])
            else:
                self.mask_lengths.append(self.sequence_length)

    # data augmentation = shifting of chords +/- 1 octave
    # there are 4 modes per chord, so we shift in multiples of 4
    def augment_sequence(self, sequence, offset):
        sequence[:, 1:128] = np.roll(sequence[:, 1:128], offset)  # melody
        sequence[:, 143:191] = np.roll(sequence[:, 143:191], offset*4)  # chord

    def process_sequence(self, seq):
        seq_chords_rhythm = seq[:, 130:]
        seq_melody = np.vstack((self.start_token, seq[:, :130]))
        
        seq_input = np.hstack((seq_melody[:-1, :], seq_chords_rhythm))
        seq_output = seq_melody[1:]
        
        return (seq_input, seq_output)
                    
    def __len__(self):
        return sys.maxsize
        
    def __getitem__(self, index):
        data_index = np.random.randint(0, len(self.data))
        data_point = self.data[data_index]
        if self.data_augmentation:
            self.augment_sequence(data_point, np.random.randint(-12, 13))
        
        data_range = np.random.randint(0, len(data_point) - self.sequence_length + 1)
        seq = data_point[data_range:(data_range + self.sequence_length)]
        
        seq_chords_rhythm = seq[:, 130:]
        seq_melody = np.vstack((self.start_token, seq[:, :130]))
        
        seq_input = np.hstack((seq_melody[:-1, :], seq_chords_rhythm))
        seq_lengths = np.asarray(self.mask_lengths[data_index], dtype=np.float32)
        seq_output = seq_melody[1:]
        
        if self.mask_lengths[data_index] < self.sequence_length:
            seq_input[self.mask_lengths[data_index], :] = 0.0
        
        # return original sequence, target sequence, and original sequence lengths
        return (
            seq_input,
            seq_output,
            seq_lengths
        )

# Sequences should be sorted decreasing in length
def sort_batch(batched_data):
    batch_size = len(batched_data[2])
    batched_data[0] = batched_data[0][[x for _,x in sorted(zip(batched_data[2], range(0, batch_size)), reverse=True)]]
    batched_data[1] = batched_data[1][[x for _,x in sorted(zip(batched_data[2], range(0, batch_size)), reverse=True)]]
    batched_data[2] = batched_data[2][[x for _,x in sorted(zip(batched_data[2], range(0, batch_size)), reverse=True)]]

def to_cuda(batched_data):
    return [x.cuda() for x in batched_data]
