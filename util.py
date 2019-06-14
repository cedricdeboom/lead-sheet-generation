import json

class Constants:
    SERVER = 1
    DATASET = "Wikifonia"
    RHYTHMS =  [0.12, 0.17, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 0.0]
    RHYTHM_TYPES =  ['32nd', '32nd', '16th', 'eighth', 'eighth', 'quarter', 'eighth', 'quarter', 'quarter', 'half', 'half', 'whole', 'barline']
    RHYTHM_EXTRAS = [None, 'dot', None, 'time-modification', None, 'time-modification', 'dot', None, 'dot', None, 'dot', None, None]
    RHYTHM_WORDS = ["32th","32th-dotted","16th", "triool", "8th", "2*triool", "8th-dotted", "quarter note", "quarter note-dotted", "half note", "half note-dotted", "whole note", "barline"]
    MIN_NEEDED_DIVISIONS = [8, 16, 4, 3, 2, 3, 4, 1, 2, 1, 1, 1, 1]
    ROOTS = ["A", "B", "C", "D", "E", "F", "G"]
    MODES = ["MAJOR", "MINOR", "AUGMENTED", "DIMINISHED"]
    MODES_WORDS = [" ", "m", " aug", " dim"]
    ALTER = [0,+1]
    SHARPS = [["A", 0], ["A", 1], ["B", 0], ["C", 0], ["C", 1], ["D", 0], ["D",1], ["E", 0], ["F",0], ["F", 1], ["G", 0], ["G", +1], ["", 0]]
    SHARPS_WORDS = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "Maatstreep"]
    PITCH_CLASSES = [["C", 0], ["C", 1], ["D", 0], ["D", 1], ["E", 0], ["F", 0], ["F", 1], ["G", 0], ["G", 1], ["A", 0], ["A", 1], ["B", 0]]

def chord_to_index(root, mode, alter):
    if not (alter == 0 or alter == 1):
        return None
    else:
        chord_symbol = root + ("#" if alter == 1 else "")
        if chord_symbol == "B#":
            chord_symbol = "C"
        elif chord_symbol == "E#":
            chord_symbol = "F"
        return (
            Constants.SHARPS_WORDS.index(chord_symbol) * len(Constants.MODES)
                + Constants.MODES.index(mode)
        )

def index_to_chord(idx):
    root = Constants.SHARPS_WORDS[idx // 4]
    mode = Constants.MODES[idx % 4]
    return (root, mode)

majors = set(['MAJOR', '', 'DOMINANT', 'MAJOR-SEVENTH', 'DOMINANT-NINTH', 'MAJOR-SIXTH', ' ', '7', 'SUSPENDED-FOURTH', 'DOMINANT-13TH', 'DOMINANT-SEVENTH', 'MAJ7', 'MAJOR-NINTH', 'DOMINANT-11TH', 'POWER', 'SUSPENDED-SECOND',  '9', '6', 'MAJOR-MINOR', 'AUGMENTED-NINTH', 'SUS47','MAJOR-13TH', 'MAJ9', 'MAJ69', '/A'])
minors = set(['MINOR', 'MINOR-SEVENTH', 'MINOR-SIXTH', 'MIN', 'MINOR-NINTH', 'MIN7', 'MINOR-11TH', 'MINOR-MAJOR', 'MIN9', 'MIN6', 'MINOR-13TH', 'MINMAJ7', 'MIN/G'])
diminisheds = set(['DIMINISHED', 'HALF-DIMINISHED', 'DIMINISHED-SEVENTH', 'DIM', 'M7B5', 'DIM7', ' DIM7'])
augmenteds = set(['AUGMENTED-SEVENTH', 'AUGMENTED', 'AUG'])

mode_translation = {}
for x in majors:
    mode_translation[x] = 'MAJOR'
for x in minors:
    mode_translation[x] = 'MINOR'
for x in diminisheds:
    mode_translation[x] = 'DIMINISHED'
for x in augmenteds:
    mode_translation[x] = 'AUGMENTED'

def pitch_to_midi_pitch(step, alter, octave):
    """!@brief Convert MusicXML pitch representation to MIDI pitch number.

        @param step Which root note it is (e.g. C, D,...)
        @param alter If the pitch was altered (sharp or flat)
        @param octave The octave that the pitch is in

        @return The MIDI pitch representation of the input
    """
    pitch_class = 0
    if step == 'C':
        pitch_class = 0
    elif step == 'D':
        pitch_class = 2
    elif step == 'E':
        pitch_class = 4
    elif step == 'F':
        pitch_class = 5
    elif step == 'G':
        pitch_class = 7
    elif step == 'A':
        pitch_class = 9
    elif step == 'B':
        pitch_class = 11
    else:
        # Raise exception for unknown step (ex: 'Q')
        raise Exception('Unable to parse pitch step ' + step)

    pitch_class = (pitch_class + int(alter)) % 12
    midi_pitch = (12 + pitch_class) + (int(octave) * 12)
    return midi_pitch

def midi_pitch_to_pitch(midi_pitch):
    if midi_pitch == 128:
        return ('rest', 0, 0)
    elif midi_pitch == 129:
        return ('barline', 0, 0)
    
    octave = midi_pitch // 12 - 1
    pitch_class = Constants.PITCH_CLASSES[midi_pitch % 12]
    step, alter = pitch_class[0], pitch_class[1]

    return (step, alter, octave)

class Entry():
    ROOT_REPLACEMENTS_FLAT = {
        'C': ('B', 0),
        'D': ('C', 1),
        'E': ('D', 1),
        'F': ('E', 0),
        'G': ('F', 1),
        'A': ('G', 1),
        'B': ('A', 1)
    }

    ROOT_REPLACEMENTS_SHARP = {
        'C': ('C', 1),
        'D': ('D', 1),
        'E': ('F', 0),
        'F': ('F', 1),
        'G': ('G', 1),
        'A': ('A', 1),
        'B': ('C', 0)
    }
    
    def __init__(self, pitch, duration, note_alter, measure, rest, root, chord_alter, mode, pos_in_measure, rhythm_index, divisions, new_measure):
        self.pitch = pitch
        self.duration = duration
        self.note_alter = note_alter
        self.measure = measure
        self.rest = rest
        self.root = root
        self.chord_alter = chord_alter
        self.mode = mode
        self.pos_in_measure = pos_in_measure
        self.rhythm_index = rhythm_index
        self.divisions = divisions
        self.new_measure = new_measure
        
        self.replaced_root = Entry.ROOT_REPLACEMENTS_FLAT[self.root][0] if self.chord_alter == -1 else self.root
        self.replaced_chord_alter = Entry.ROOT_REPLACEMENTS_FLAT[self.root][1] if self.chord_alter == -1 else self.chord_alter

        self.replaced_root_2 = Entry.ROOT_REPLACEMENTS_SHARP[self.replaced_root][0] if self.replaced_chord_alter == 1 else self.replaced_root
        self.replaced_chord_alter_2 = Entry.ROOT_REPLACEMENTS_SHARP[self.replaced_root][1] if self.replaced_chord_alter == 1 else self.replaced_chord_alter

        self.replaced_root = self.replaced_root_2
        self.replaced_chord_alter = self.replaced_chord_alter_2

        del self.replaced_root_2
        del self.replaced_chord_alter_2
        
    def __str__(self):
        n = f"NOTE  {self.pitch}({self.note_alter}), dur {self.duration}/{self.divisions}, meas {self.measure}, new? {self.new_measure}"
        c = f"CHORD {self.replaced_root}({self.replaced_chord_alter}), r {self.rhythm_index}, mode {self.mode}, pos {self.pos_in_measure}"
        return n + "\n\t" + c
    
    def to_dict(self):
        ret = {}
        ret['note'] = {'pitch': self.pitch, 'alter': self.note_alter, "rest": self.rest}
        ret['rhythm'] = {'duration': self.duration, 'divisions': self.divisions, 'index': self.rhythm_index, 'new_measure': self.new_measure}
        ret['chord'] = {'root': self.replaced_root, 'mode': self.mode, 'alter': self.replaced_chord_alter}
        return ret
