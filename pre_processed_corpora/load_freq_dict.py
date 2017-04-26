import os
import json
from pre_processed_corpora.load_pre_processed_corpus import load_child_directed_speech


file_dir = os.path.dirname(os.path.realpath(__file__))


def create_freq_dict(corpus_name, path_):

    utterances = load_child_directed_speech(corpus_name)

    freq_dict = dict()
    for u in utterances:
        for w in u:
            if w not in freq_dict:
                freq_dict[w] = 0
            freq_dict[w] += 1

    json.dump(freq_dict, open(path_, 'w'))


def load_freq_dict(corpus_name):
    """load freq dict from disk and return it if it exists; else create it, save it to disk, and return it"""
    path_ = os.path.join(file_dir, '%s_freq_dict.json' % corpus_name)

    try:
        freq_dict = json.load(open(path_, 'r'))
    except FileNotFoundError:
        create_freq_dict(corpus_name, path_)
        freq_dict = json.load(open(path_, 'r'))

    return freq_dict
