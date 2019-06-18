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
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers_lstm=3, inference=False):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers_lstm = num_layers_lstm
        self.inference = inference
        
        self.hidden = self.init_hidden()
        
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.num_layers_lstm,
            batch_first=True,
            bidirectional=False
        )
        
        # Last dense layer
        self.dense = nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (
            (torch.zeros(self.num_layers_lstm, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers_lstm, self.batch_size, self.hidden_dim)),
            (torch.zeros(self.num_layers_lstm, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers_lstm, self.batch_size, self.hidden_dim)),
            (torch.zeros(self.num_layers_lstm, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers_lstm, self.batch_size, self.hidden_dim))
            )

    def forward(self, input_X, lengths_X, max_length, temperature=1.0, propagate_hidden=False):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).
        
        # get chord and rhythm info from input
        if not self.inference:
            packed_X = torch.nn.utils.rnn.pack_padded_sequence(input_X, lengths_X, batch_first=True)
        else:
            packed_X = input_X
        
        lstm_out, self.hidden = self.lstm(packed_X)
        
        if not self.inference:
            unpacked_X, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=max_length)
        else:
            unpacked_X = lstm_out
        
        # Generate sequence predictions
        X_pred = self.dense(unpacked_X)
        
        # Apply nonlinearities
        X_softmax_1 = F.log_softmax(X_pred[:, :, :130] / temperature, dim=2)
        X_softmax_2 = F.log_softmax(X_pred[:, :, 130:143] / temperature, dim=2)
        X_softmax_3 = F.log_softmax(X_pred[:, :, 143:] / temperature, dim=2)
        
        # Concatenate the two tensors along axis 2
        X_return = torch.cat((X_softmax_1, X_softmax_2, X_softmax_3), 2)
        
        return X_return


# NUMPY INPUT DATA SCHEMA
# [0:127] -> MIDI pitch (one-hot) - melody
# [128] -> rest (one-hot) - melody
# [129] -> barline (one-hot) - melody
# [130:141] -> rhythm index (one-hot) - rhythm
# [142] -> barline (one-hot) (repeated) - rhythm
# [143:190] -> chord (one-hot) - chord
# [191] -> barline (one-hot) (repeated) - chord

class MusicDataset(Dataset):
    def __init__(self, data_file, sequence_length, data_augmentation, use_len_data=False):
        super(MusicDataset, self).__init__()
        self.sequence_length = sequence_length
        self.data_augmentation = data_augmentation
        self.use_len_data = use_len_data
        
        with open(data_file, 'rb') as f:
            self.id_to_sheet = pickle.load(f)
            self.data = pickle.load(f)
        
        self.data = [x.astype(np.float32) for x in self.data]
        self.start_token = np.zeros((1, 130), dtype=np.float32)
        self.start_token[:, 0] = 1.0
        
        # pad all sequences to desired sequence length
        self.mask_lengths = []
        for i, x in enumerate(self.data):
            if len(x) < self.sequence_length + 1:
                s = x.shape
                self.data[i] = np.zeros((self.sequence_length + 1, s[1]), dtype=np.float32)
                self.data[i][:s[0], :] = x
                self.mask_lengths.append(s[0] - 1)
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
        if self.use_len_data:
            return len(self.data)
        else:
            return sys.maxsize
        
    def __getitem__(self, index):
        data_index = np.random.randint(0, len(self.data))
        data_point = self.data[data_index]
        if self.data_augmentation:
            self.augment_sequence(data_point, np.random.randint(-12, 13))
        
        data_range = np.random.randint(0, len(data_point) - self.sequence_length)
        
        # return original sequence, target sequence, and original sequence lengths
        return (
            data_point[data_range:(data_range + self.sequence_length)],
            data_point[data_range + 1:(data_range + self.sequence_length + 1)],
            np.asarray(self.mask_lengths[data_index], dtype=np.float32)
        )

# Sequences should be sorted decreasing in length
def sort_batch(batched_data):
    batch_size = len(batched_data[2])
    batched_data[0] = batched_data[0][[x for _,x in sorted(zip(batched_data[2], range(0, batch_size)), reverse=True)]]
    batched_data[1] = batched_data[1][[x for _,x in sorted(zip(batched_data[2], range(0, batch_size)), reverse=True)]]
    batched_data[2] = batched_data[2][[x for _,x in sorted(zip(batched_data[2], range(0, batch_size)), reverse=True)]]

def to_cuda(batched_data):
    return [x.cuda() for x in batched_data]
