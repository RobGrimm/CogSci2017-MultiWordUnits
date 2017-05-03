import os
import pickle


file_dir = os.path.dirname(os.path.realpath(__file__))


def load_mwus(corpus_name):
    mwus = pickle.load(open(os.path.join(file_dir, '%s.pickle' % corpus_name), 'rb'))
    return mwus