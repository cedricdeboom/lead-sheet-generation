import numpy as np
import json
import os
import sys
import pickle

import util
from util import Constants

DATA_DIR = 'chord_rhythm_lstm/processed_output'
OUT_FILE = 'chord_rhythm_lstm/processed_output_numpy/data.pkl'
DIMENSIONS = 192

# SHEET SCHEMA
# {'note': {'pitch': int, 'alter': int, 'rest': boolean},
#   'rhythm': {'duration': float,
#    'divisions': float,
#    'index': int,
#    'new_measure': boolean},
#   'chord': {'root': String, 'mode': String, 'alter': int}}

# NUMPY SCHEMA
# [0:127] -> MIDI pitch (one-hot) - melody
# [128] -> rest (one-hot) - melody
# [129] -> barline (one-hot) - melody
# [130:141] -> rhythm index (one-hot) - rhythm
# [142] -> barline (one-hot) (repeated) - rhythm
# [143:190] -> chord (one-hot) - chord
# [191] -> barline (one-hot) (repeated) - chord

all_np_sheets = []
index_to_title = {}

index = 0
for sheet in sorted(os.listdir(DATA_DIR)):
    print(index, end='\r')
    if not sheet.endswith('.json'):
        continue
    index_to_title[index] = sheet
    json_file = os.path.join(DATA_DIR, sheet)

    np_sheet = []
    try:
        with open(json_file, 'r') as f:
            sheet_json = json.load(f)

            for x in sheet_json:
                if x['rhythm']['new_measure']:
                    entry = np.zeros(DIMENSIONS)
                    entry[129] = 1
                    entry[142] = 1
                    entry[191] = 1
                    np_sheet.append(entry)
                
                entry = np.zeros(DIMENSIONS)
                
                if not x['note']['rest']:
                    entry[0 + x['note']['pitch']] = 1
                entry[128] = int(x['note']['rest'])
                entry[130 + x['rhythm']['index']] = 1
                
                chord_index = util.chord_to_index(x['chord']['root'], x['chord']['mode'], x['chord']['alter'])
                entry[143 + chord_index] = 1
                
                np_sheet.append(entry)
        
        if len(np_sheet) > 5:
            all_np_sheets.append(np.asarray(np_sheet))
            index += 1
    except:
        print("\n", sheet, " : UNSUPPORTED ALTER IN CHORD")


with open(OUT_FILE, 'wb') as f:
    pickle.dump(index_to_title, f)
    pickle.dump(all_np_sheets, f)
