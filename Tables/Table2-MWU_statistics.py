import numpy as np
import scipy.stats
from ChunkBasedLearner.load_mwus import load_mwus as load_cbl_mwus
from PredictionBasedSegmenter.load_mwus import load_mwus as load_pbs_mwus


def print_mwu_info(mwu_type):

    for corpus_name in ['NA', 'BE']:

        if mwu_type == 'CBL':
            mwu_counter = load_cbl_mwus(corpus_name)
        elif mwu_type == 'PBS':
            mwu_counter = load_pbs_mwus(corpus_name)
        else:
            raise Exception('mwu_type must be one of: CBL, PBS')

        mwus = list(mwu_counter.keys())
        mwu_lengths = [len(i) for i in mwus]
        nr_mwu_types = len(mwus)
        nr_mwu_tokens = sum([mwu_counter[i] for i in mwus])

        print(corpus_name, mwu_type)
        print('nr MWU tokens: %s' % nr_mwu_tokens)
        print('nr MWU types: %s' % nr_mwu_types)
        print('median MWU length: %s (IQR: %s)' % (np.median(mwu_lengths), scipy.stats.iqr(mwu_lengths)))
        print()


if __name__ == '__main__':
    print_mwu_info(mwu_type='CBL')
    print_mwu_info(mwu_type='PBS')