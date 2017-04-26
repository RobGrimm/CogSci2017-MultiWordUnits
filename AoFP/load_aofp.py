import os
import json


file_dir = os.path.dirname(os.path.realpath(__file__))


def load_age_of_first_production(cds_corpus):
    """ load and return AoFP dictionary """
    aoa_dict = json.load(open('/'.join([file_dir, '%s.json' % cds_corpus]), 'r'))
    return aoa_dict