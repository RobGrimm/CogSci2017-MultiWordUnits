import os
import json
from CHILDES.pre_process_childes import process_childes_corpora

file_dir = os.path.dirname(os.path.realpath(__file__))

########################################################################################################################

# 1. pre-process British English corpora


british_corpora = ['Belfast', 'Fletcher', 'Manchester', 'Thomas', 'Tommerdahl', 'Wells', 'Forrester', 'Lara']

# process CHILDES .xml files
corpus_dict = process_childes_corpora(corpora=british_corpora, corpora_dir='corpora/BE/', MLU=True)

# write processed corpus data to disk for future usage
json.dump(corpus_dict, open(os.path.join(file_dir, 'pre_processed_corpora', 'BE.json'), 'w'))



########################################################################################################################

# 2. pre-process American English corpora


american_corpora = ['Bates', 'Bernstein', 'Bliss', 'Bloom70', 'Bloom73', 'Bohannon', 'Braunwald', 'Brent', 'Brown',
                    'Carterette', 'Clark', 'Cornell', 'Demetras1', 'Demetras2', 'ErvinTripp', 'Evans', 'Feldman',
                    'Garvey', 'Gathercole',  'Gleason', 'HSLLD', 'Hall', 'Higginson', 'Kuczaj', 'MacWhinney', 'McCune',
                    'McMillan',  'Morisset', 'Nelson', 'NewEngland', 'Peters', 'Post', 'Providence', 'Rollins', 'Sachs',
                    'Snow', 'Soderstrom', 'Sprott', 'Suppes', 'Tardif', 'Valian', 'VanHouten', 'VanKleeck', 'Warren',
                    'Weist']

# process CHILDES .xml files
corpus_dict = process_childes_corpora(corpora=american_corpora, corpora_dir='corpora/NA/', MLU=True)

# write processed corpus data to disk for future usage
json.dump(corpus_dict, open(os.path.join(file_dir, 'pre_processed_corpora', 'NA.json'), 'w'))