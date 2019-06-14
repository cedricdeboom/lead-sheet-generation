
import xml.etree.ElementTree as ET
import re
import copy


class DataAugmentationAndPreprocessing:

    def __init__(self):
        print("---------------------------------------------------------------")
        print("----------- DATA AUGMENTATION AND PREPROCESSING ---------------")

    def do_preprocessing(self, scores):
        """!@brief This function does all the preprocessing steps on the piece, 
        and returns the new piece

        @param scores The scores in array format in ElementTree format

        @return The preprocessed scores in ElementTree format
        """
        # Preprocessing
        print("--------------------- BEGIN PREPROCESSING ---------------------")
        new_scores = []
        for piece in scores:
            piece = self.erase_double_notes(piece)
            piece = self.flatten(piece) #TODO find mistake in flatten that divisions isn't anymore in the piece
            new_scores.append(piece)

        return new_scores

    def do_data_augmentation(self, music_reps):
        """!@brief This function does all the data augmentation steps on the piece, 
        and returns the new music_representations

        @param music_reps the scores in Music_representation format

        @return The new data-augmented music_representation scores
        """
        print("------------------- BEGIN DATA AUGMENTATION -------------------")
        music_reps_new = []
        for score in music_reps:
            new_scores = self.transpose(score)
            for new_score in new_scores:
                music_reps_new.append(new_score)

        # print(music_reps_new)
        print("-------------------- END DATA AUGMENTATION --------------------")
        return music_reps_new



    # PREPROCESSING
    def erase_double_notes(self, piece):
        """!@brief PREPROCESSING
            This function will erase the double notes from a piece. 

            @param piece ElementTree of a score
            
            @return piece without the double notes
        """

        # iterate over the measures
        for measure in piece.iter('measure'):
            notes_time = {}
        
            # if note has same default-x in measure => double notes
            for note in measure.iter('note'):
                if 'default-x' in note.attrib:
                    attrib = note.attrib['default-x']
                    if notes_time.get(attrib, 0) == 0:
                        notes_time[attrib] = [note]
                    else:
                        notes_time[attrib].append(note)


            # remove the notes
            for note_arr in notes_time.values(): 
                if len(note_arr) > 1:
                    pitch_compare = -float("infinity")
                    highest_note = None
                    notes_to_delete = []

                    for note in note_arr:
                        pitch = note.find('pitch')
                        if pitch != None:
                            # it's not a rest
                            step = pitch.find('step').text
                            octave = pitch.find('octave').text
                            alter = pitch.find('alter')
                            if alter != None:
                                alter = alter.text
                            else:
                                alter = 0

                            # Compute MIDI pitch number (C4 = 60, C1 = 24, C0 = 12)
                            pitch_midi = self.pitch_to_midi_pitch(step, alter, octave)

                            # Keep only highest note
                            if pitch_midi > pitch_compare:
                                pitch_compare = pitch_midi
                                if highest_note is not None:
                                    notes_to_delete.append(highest_note)
                                highest_note = note
                            else:
                                notes_to_delete.append(note)
                        
                    # remove the notes
                    for note in notes_to_delete:
                        measure.remove(note)
                            
                        

        return piece
        

    # PREPROCESSING
    def flatten(self, piece):
        """!@brief This function will make sure that the repeat markers are deleted and 
        instead the piece is unrolled

        @param piece ElementTree of piece

        @return the flattened piece

        Examples of the repeats in the mxl format of ElementTree:

        @code The following markers are all under the <measure> tag of the mxl files

        LEFT REPEAT BAR ||:
            <barline location="left">
                <bar-style>heavy-light</bar-style>
                <repeat direction="forward"/>
            </barline>

        RIGHT REPEAT BAR :||
        <barline location="right">
            <bar-style>light-heavy</bar-style>
            <repeat direction="backward"/>
        </barline>

        LEFT HEADER ----1-----
        <barline location="right">
            <bar-style>light-heavy</bar-style>
            <ending print-object="yes" type="stop" number="1"></ending>
            <repeat direction="backward"/>
        </barline>

        RIGHT HEADER ---2----
        <barline location="right">
            <bar-style>light-light</bar-style>
            <ending type="stop" number="2"></ending>
        </barline>
        """
        title = piece.find('movement-title')
        # if title is not None:
        #     print(title.text)
        # else:
        #     print("GEEN TITEL")
        structure, zeroMeasure = self.find_structure_piece(piece)
        if zeroMeasure: # opmaat
            structure = [0] + structure
        # print(structure)

        measures = []
        # find all the measures
        for i in structure:
            which_measure = './/measure[@number="'+str(i)+'"]'
            measure = piece.find('part').findall(which_measure)
            measures.append(measure)



        # delete the regular measures
        for m in piece.find('part').findall('.//measure'):
            piece.find('part').remove(m)

        # add the new structure
        for m in measures:
            if len(m) > 0:
                piece.find('part').append(m[0])



        return piece

    def find_structure_piece_help_function(self, piece):
        """!@brief This help function takes a piece and finds the forward direction measures, backward direction measure, 
        The measures where there is a one above (repetition structure), the measures where there is a two or three above as well
        Also has some other vars that help out for the main function.
        """
        # output
        forward_direction = [] # ||:
        backward_direction = [] # :||
        one_measures = [] # ----1-----
        two_measures = [] # ----2-----
        three_measures = [] # ----3----
        same = True
        zeroMeasure = False

        # iterate over all measures
        number_of_measures = 0
        for measure in piece.iter('measure'):
            measure_number = measure.attrib['number']
            measure_number2 = re.sub("[^0-9]", "", measure_number)
            if measure_number2 != measure_number:
                same = False
            # measure_number = re.sub("[^0-9]", "", measure_number)
            if measure_number != '0':
                number_of_measures += 1

                # find the attribbute bar line
                for barline in measure.iter('barline'): 
                    if barline != None and measure_number != '':
                        string = ET.tostring(piece, encoding='utf8', method='xml')

                        # find repeat and ending tags
                        repeat = barline.find('repeat') 
                        ending = barline.find('ending') 
                        
                        # print(measure.attrib['number'])
                        # print(repeat)

                        # if repeat tag exists, find forward and backward measures numbers and append to output
                        if repeat != None:
                            # print(ET.tostring(measure))
                            direction = repeat.attrib['direction']
                            if direction == 'forward':
                                forward_direction.append(measure_number)
                            elif direction == 'backward':
                                backward_direction.append(measure_number)
                        
                        # if ending tag exists, find 1 or 2
                        if ending != None:
                            # print(ET.tostring(measure))
                            number = ending.attrib['number']
                            if number == '1':
                                one_measures.append(measure_number)
                            elif number == '1, 2'or number == '1,2'or number == '1 , 2':
                                one_measures.append(measure_number)
                                two_measures.append(measure_number)
                            elif number == '2':
                                two_measures.append(measure_number)
                            elif number == '3':
                                three_measures.append(measure_number)
            else: #Opmaat
                zeroMeasure = True
                                
        return forward_direction, backward_direction, one_measures, two_measures, three_measures, same, number_of_measures, zeroMeasure


    def find_structure_piece(self, piece):
        """!@brief Return the structure of the piece, how it can be unfolded"""
        forward_direction, backward_direction, one_measures, two_measures, three_measures, same, number_of_measures, zeroMeasure = self.find_structure_piece_help_function(piece)

        structure = []

        if same:
            # make the values unqiue
            forward_direction = list(set(forward_direction))
            backward_direction = list(set(backward_direction))
            one_measures = list(set(one_measures))
            two_measures = list(set(two_measures))
            three_measures = list(set(three_measures))


            # sort 
            forward_direction.sort(key=float)
            backward_direction.sort(key=float)
            one_measures.sort(key=float)
            two_measures.sort(key=float)
            three_measures.sort(key=float)

            one = None

        
            # if there is a repeat
            if backward_direction:
                # find structure

                last = 0 # the last measure that was added
                for test in range(0,len(backward_direction)):
                    if len(backward_direction) > 0:
                        i = int(backward_direction.pop(0))
                        # print("BACK: "+ str(i))
                        
                        # add the first measures
                        for j in range(last, i):
                            structure.append(j+1)

                        
                        begin = None
                        # if there is a ||:
                        if len(forward_direction) > 0 and int(forward_direction[0]) < i:
                            begin = int(forward_direction.pop(0))
                        # no forward repeat, therefore beginning
                        else:
                            begin = last+1
                        # print("BEGIN: " + str(begin))

                        if len(one_measures) > 0 and int(one_measures[0]) <= i:
                            one = int(one_measures.pop(0))
                            # print("ONE: "+str(one))
                            # add the second repeat
                            for j in range(begin,one):
                                structure.append(j) 
                        else:
                            for j in range(begin, i+1):
                                structure.append(j)

                        last = i 

                        if len(three_measures) > 0:
                            if len(backward_direction) > 0:
                                if int(three_measures[0]) == int(backward_direction[0])+1:
                                    last = int(backward_direction.pop(0))
                                    if len(two_measures) > 0:
                                        two = int(two_measures.pop(0))
                                    else: 
                                        two = last
                                    three = int(three_measures.pop(0))
                                    for j in range(two, three):
                                        structure.append(j)
                                    # add the third repeat
                                    if one == None:
                                        one = last
                                    for j in range(begin,one):
                                        structure.append(j)
                                    

                                    if len(backward_direction) == 0:
                                        break
                            else:
                                if int(three_measures[0]) == last+1:
                                    if len(two_measures) > 0:
                                        two = int(two_measures.pop(0))
                                    else: 
                                        two = last
                                    three = int(three_measures.pop(0))
                                    for j in range(two, three):
                                        structure.append(j)
                                    # add the third repeat
                                    if one == None:
                                        one = last
                                    for j in range(begin,one):
                                        structure.append(j)
                                    

                                    if len(backward_direction) == 0:
                                        break


                    # print(structure)
            # no repeat was found
            else: 
                for i in range(0,number_of_measures):
                    structure.append(i+1)

            # the last measures
            if structure[-1] < number_of_measures: 
                for i in range(last, number_of_measures):
                    structure.append(i+1)
        else:
            for measure in piece.iter('measure'):
                measure_number = measure.attrib['number']
                structure.append(measure_number)


        # print(structure)
        return structure, zeroMeasure

    
    @staticmethod
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


if __name__ == '__main__':
    print('BEGIN')
    # mxl = DataAugmentationAndPreprocessing("/Users/stephanievanlaere/Documents/Dataset/Lead Sheets/Small")
