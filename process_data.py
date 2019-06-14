import xml.etree.ElementTree as ET
import zipfile
import argparse
import json
import os
import re
import six

import util
from data_augmentation import DataAugmentationAndPreprocessing


#DATA_DIR = 'data/Wikifonia/'
DATA_DIR = 'chord_rhythm_lstm/output'
OUT_DIR = 'chord_rhythm_lstm/processed_output'
da = DataAugmentationAndPreprocessing()


parser = argparse.ArgumentParser(description='Options')
parser.add_argument('processing_seeds', type=bool,
                help='Required if processing seeds or not')

args = parser.parse_args()

###
# 1. Get all lead sheets
###
sheet_names = [x for x in os.listdir(DATA_DIR) if x.endswith('.mxl')]

###
# 2. Process MXL for each sheet
###

for sheet in sheet_names:
    print(sheet)
    processed_entries = []
    start_of_piece = True
    dont_output = False

    try:
        sheet_file = zipfile.ZipFile(os.path.join(DATA_DIR, sheet))
    except:
        continue
    infolist = sheet_file.infolist()

    # Python 3 UTF-8 encoding error workaround
    if six.PY3:
        # https://stackoverflow.com/questions/37723505/namelist-from-zipfile-returns-strings-with-an-invalid-encoding
        zip_filename_utf8_flag = 0x800
        for info in infolist:
            if info.flag_bits & zip_filename_utf8_flag == 0:
                filename_bytes = info.filename.encode('437')
                filename = filename_bytes.decode('utf-8', 'replace')
                info.filename = filename
                
    container = [x for x in infolist if 'container.xml' in x.filename][0]

    compressed_file_name = ''
    try:
        container_parsed = ET.fromstring(sheet_file.read(container))
        for rootfile_tag in container_parsed.findall('./rootfiles/rootfile'):
            compressed_file_name = rootfile_tag.attrib['full-path']
    except ET.ParseError as exception:
        print(f'SKIPPING {sheet} due to MXL parsing error')

    compressed_file_info = [x for x in infolist if x.filename == compressed_file_name][0]
    score_string = sheet_file.read(compressed_file_info)
    score = ET.fromstring(score_string)
    score = da.do_preprocessing([score])[0]

    chord_prev_root = None
    chord_to_play = None

    for measure in score.iter('measure'):
        new_measure = True
        
        # Get Divisions
        for division in measure.iter('divisions'):
            if division != None:
                divisions = float(division.text)
            else:
                divisions = 8
        
        # Get bar number
        measure_number = measure.attrib['number']
        if measure_number != '':
            measure_number = int(re.sub("[^0-9]", "", measure_number))
        else:
            measure_number = 0
        
        # Get Time Signature
        # find the time bar line and if exists, add Time Signature to score
        times = measure.iter('time')
        for time in times:
            if time != None:
                bar_number = measure_number
                numerator = int(time.find('beats').text)
                denominator = int(time.find('beat-type').text)
                
        # Get Key Signature
        keys = measure.iter('key')
        for key in keys:
            if key != None:
                # set fifth of key signature, won't show up in print if it's zero
                fifths = int(key.find('fifths').text)
                if fifths is None:
                    fifths = 0

                # mode, won't show up in print if it's not major or minor
                if key.find('mode'):
                    mode = key.find('mode').text.upper()
                    if mode == "MAJOR":
                        mode_num = 1
                    elif mode == "MINOR":
                        mode_num = 2
                    else:
                        mode_num = 0
                else:
                    mode_num = 0
            
        # Get Tempo
        sounds = measure.iter('sound')
        for sound in sounds:
            if sound != None:
                if 'tempo' in sound.attrib:
                    tempo = sound.attrib['tempo']
                    qpm = float(tempo)
                    
        # see how the chord look in the measure
        meas_chords = []
        how_far_in_measure = 0
        for item in measure:
            if item.tag == 'harmony':
                # new chord
                offset = 0
                if item.find('offset') is not None:
                    try:
                        offset = int(float(item.find('offset').text))
                    except ValueError:
                        print(item.find('offset').text)
                        print('value error')


                ## First up: Root and Root alter
                # Root
                root = item.find('root')
                if root is not None:
                    root_step = root.find('root-step').text
                    root_alter = root.find('root-alter')

                    # alter
                    if root_alter is not None:
                        alter = int(root_alter.text)
                    else:
                        alter = 0

                    ## Second up: Mode
                    mode = item.find('kind')
                    if mode is not None:
                        mode = str(mode.text).upper()

                        # if major mode
                        if mode in util.mode_translation:
                            mode = util.mode_translation[mode]
                        else:
                            print('New mode was found that is not in one of the arrays')
                            print(mode)
                    else:
                        mode = "MAJOR"

                    # Position in measure
                    length = how_far_in_measure + offset
                else:
                    print("ROOT IS NONE")

                meas_chords.append({"root": root_step, "alter": alter, "mode": mode, "pos_in_measure": length})

            if item.tag == 'note':
                duration = item.find('duration')
                if duration is not None:
                    duration = int(duration.text)
                    how_far_in_measure = how_far_in_measure + duration
                    
        # Notes
        notes = measure.iter('note')
        current_position_in_measure = 0 # that's how far we are the measure
        old_duration = 0 # that's the previous duration of notes

        for note in notes:
            rest = False
            note_duration = 0.0
            note_measure = 0
            note_pitch = 0
            note_alter = 0
            entry = None

            if note != None:
                duration = note.find('duration')
                if duration != None: # anders versiernoten die we niet gebruiken
                    # see how the measure looks like note wise
                    current_position_in_measure = current_position_in_measure + old_duration
                    old_duration = float(duration.text)  

                    if len(meas_chords) > 0 and measure_number != 0: # opmaatnoten niet meerekennen
                        ## FIRST UP: THE Note

                        # if there is a pitch, otherwise rest
                        pitch = note.find('pitch')
                        if pitch != None:
                            # it's not a rest
                            note_rest = False
                            step = pitch.find('step').text
                            octave = pitch.find('octave').text
                            alter = pitch.find('alter')
                            if alter != None:
                                alter = alter.text
                            else:
                                alter = 0

                            # Compute MIDI pitch number (C4 = 60, C1 = 24, C0 = 12)
                            pitch_midi = util.pitch_to_midi_pitch(step, alter, octave)
                            note_pitch = pitch_midi
                        else:
                            note_rest = True

                        # Duration
                        note_duration = float(duration.text)

                        # Measure
                        note_measure = False
                        
                        note_rep = (note_pitch, note_duration, note_alter, note_measure)

                        if len(meas_chords) > 0: # Als er akkoorden nog zijn in de meas_chords,
                                                # anders gebruik het vorig akkoord. Opmaten worden niet meegeteld.

                            ## SECOND UP: THE Chord
                            # Find the chord to be played during the note
                            dur = [i["pos_in_measure"] for i in meas_chords]

                            if chord_prev_root is None:
                                chords_possible_to_play = [dur[0]] if len(dur) > 0 else []
                            else:
                                chords_possible_to_play = [x for x in dur if x <= current_position_in_measure]
                            if len(chords_possible_to_play) > 0:
                                chord_to_play_during_note = max(chords_possible_to_play)
                                index_of_chord_to_play_during_notes = dur.index(chord_to_play_during_note)
                                chord_to_play = meas_chords[index_of_chord_to_play_during_notes]

                                # Set the values
                                chord_root = chord_to_play["root"]
                                chord_alter = chord_to_play["alter"]
                                chord_mode = chord_to_play["mode"]
                                duration_chord = float(float(note_duration)/float(divisions))
                                duration_chord = float("{0:.2f}".format(duration_chord))
                                if duration_chord in util.Constants.RHYTHMS:
                                    chord_rhythm = util.Constants.RHYTHMS.index(duration_chord)
                                else:
                                    print("Rhythm not right")
                            elif chord_prev_root != None:
                                chord_root = chord_prev_root
                                chord_alter = chord_prev_alter
                                chord_mode = chord_prev_mode
                                duration_chord = float(float(note_duration)/float(divisions))
                                duration_chord = float("{0:.2f}".format(duration_chord))
                                if duration_chord in util.Constants.RHYTHMS:
                                    chord_rhythm = util.Constants.RHYTHMS.index(duration_chord)
                                else:
                                    print("**************** Rhythm not right")
                            else:
                                print("**************** WHUUUUTT")
                                dont_output = True


                            # only if in the measure there is no chord, we use this one
                            if chord_root != "":
                                chord_prev_root = chord_root
                                chord_prev_alter = chord_alter
                                chord_prev_mode = chord_mode

                    elif measure_number != 0: # Geen akkoord in de maat zelf, moet het vorig akkoord gebruiken
                        if chord_prev_root != None:
                            ## FIRST UP: THE Note
                            # if there is a pitch, otherwise rest
                            pitch = note.find('pitch')
                            if pitch != None:
                                # it's not a rest
                                note_rest = False
                                step = pitch.find('step').text
                                octave = pitch.find('octave').text
                                alter = pitch.find('alter')
                                if alter != None:
                                    alter = alter.text
                                else:
                                    alter = 0

                                # Compute MIDI pitch number (C4 = 60, C1 = 24, C0 = 12)
                                pitch_midi = util.pitch_to_midi_pitch(step, alter, octave)
                                note_pitch = pitch_midi
                            else:
                                note_rest = True

                            # Duration
                            note_duration = float(duration.text)

                            # Measure
                            note_measure = False
                            
                            note_rep = (note_pitch, note_duration, note_alter, note_measure)

                            ## SECOND UP: CHORD
                            chord_root = chord_prev_root
                            chord_alter = chord_prev_alter
                            chord_mode = chord_prev_mode
                            duration_chord = float(float(note_duration)/float(divisions))
                            duration_chord = float("{0:.2f}".format(duration_chord))
                            if duration_chord in util.Constants.RHYTHMS:
                                chord_rhythm = util.Constants.RHYTHMS.index(duration_chord)
                            else:
                                print("**************** Rhythm not right")
                    
                    entry = None
                    if chord_to_play is not None:
                        entry = util.Entry(note_pitch, note_duration, note_alter, note_measure, note_rest, chord_to_play['root'], chord_to_play['alter'], chord_to_play['mode'], chord_to_play['pos_in_measure'], chord_rhythm, divisions, new_measure)
                    else:
                        entry = util.Entry(note_pitch, note_duration, note_alter, note_measure, True, "", 0, "", 0, 0, divisions, new_measure)
                        
                    if (start_of_piece and entry.pitch != 0) or args.processing_seeds:
                        start_of_piece = False
                    
                    if not start_of_piece:
                        processed_entries.append(entry)

                    if new_measure:
                        new_measure = False
    
    if not dont_output:
        with open(os.path.join(OUT_DIR, sheet[:-4] + '.json'), 'w') as f:
            json.dump([x.to_dict() for x in processed_entries], f)
