import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import xml.etree.ElementTree as ET
import zipfile
import six
import numpy as np
import util
import shutil
import math

class ChordRhythmMXL():
    CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
    <container>
        <rootfiles>
            <rootfile full-path="sheet.xml"></rootfile>
        </rootfiles>
    </container>
"""

    def __init__(self, xml_template):
        self.xml_template = xml_template
        self.sheet = ET.parse(xml_template)
        self.divs = None

    def write_mxl(self, mxl_output_file):
        if os.path.isdir("zip_tmp"):
            shutil.rmtree("zip_tmp")
        os.mkdir("zip_tmp")

        # write the xml file
        self.sheet.write("zip_tmp/sheet.xml")

        # write meta
        os.mkdir("zip_tmp/META-INF")
        with open('zip_tmp/META-INF/container.xml', 'w') as f:
            f.writelines(ChordRhythmMXL.CONTAINER)
        
        # zip the whole
        shutil.make_archive(mxl_output_file, 'zip', 'zip_tmp')
        shutil.rmtree("zip_tmp")
        os.rename(mxl_output_file + '.zip', mxl_output_file)

    def process(self, data):
        # STEP 1: find divisions
        needed_divisions = [util.Constants.MIN_NEEDED_DIVISIONS[r[0]] for r in data]

        def find_lcm(x):
            lcm = needed_divisions[0]
            for i in needed_divisions[1:]:
                lcm = lcm * i // math.gcd(lcm, i)
            return lcm

        self.divs = find_lcm(needed_divisions)

        # STEP 2: fill measures
        part_xml = self.sheet.getroot().find('part')
        measure_count = 0
        for idx, d in enumerate(data):
            if d[0] == 12 and d[1] == 48:
                if measure_count >= 1:
                    part_xml.append(measure)
                
                measure = ET.Element('measure')
                measure_count += 1
                measure.attrib['number'] = str(measure_count)
                
                if measure_count == 1:
                    attributes = ET.Element('attributes')
                    divisions = ET.Element('divisions')
                    divisions.text = str(self.divs)
                    attributes.append(divisions)
                    measure.append(attributes)
                
            else:
                root, mode = util.index_to_chord(d[1])
                alter = 0
                if '#' in root:
                    root = root[:-1]
                    alter = 1
                duration = int(round(util.Constants.RHYTHMS[d[0]] * self.divs))
                
                harmony = ET.Element('harmony')
                harmony.attrib['print-frame'] = 'no'
                
                root_elem = ET.Element('root')
                rootstep_elem = ET.Element('root-step')
                rootstep_elem.text = root
                root_elem.append(rootstep_elem)
                if alter != 0:
                    rootalter_elem = ET.Element('root-alter')
                    rootalter_elem.text = str(alter)
                    root_elem.append(rootalter_elem)
                harmony.append(root_elem)
                
                kind_elem = ET.Element('kind')
                kind_elem.text = mode.lower()
                harmony.append(kind_elem)
                
                measure.append(harmony)
                
                note = ET.Element('note')
                
                rest_elem = ET.Element('rest')
                rest_elem.text = ''
                note.append(rest_elem)
                
                dur_elem = ET.Element('duration')
                dur_elem.text = str(duration)
                note.append(dur_elem)
                
                voice_elem = ET.Element('voice')
                voice_elem.text = str(1)
                note.append(voice_elem)
                
                type_elem = ET.Element('type')
                type_elem.text = util.Constants.RHYTHM_TYPES[d[0]]
                note.append(type_elem)
                
                extra = util.Constants.RHYTHM_EXTRAS[d[0]]
                if extra:
                    if extra == 'dot':
                        dot_elem = ET.Element('dot')
                        dot_elem.text = ''
                        note.append(dot_elem)
                    elif extra == 'time-modification':
                        timemod_elem = ET.Element('time-modification')
                        actual_elem = ET.Element('actual-notes')
                        actual_elem.text = str(3)
                        timemod_elem.append(actual_elem)
                        
                        normal_elem = ET.Element('normal-notes')
                        normal_elem.text = str(2)
                        timemod_elem.append(normal_elem)
                        
                        note.append(timemod_elem)
                
                measure.append(note)
                
        if len(measure.getchildren()) > 0:
            part_xml.append(measure)
