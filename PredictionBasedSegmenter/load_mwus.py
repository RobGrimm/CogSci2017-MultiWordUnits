import os


file_dir = os.path.dirname(os.path.realpath(__file__))


def load_mwus(corpus_name):

    mwu_counter = dict()

    with open(os.path.join(file_dir, '%s.txt' % corpus_name), 'r') as f:
        for line in f:
            line = line.split()
            mwu = tuple(line[:-1])
            if len(mwu) == 1: # do not consider single words
                continue
            freq = float(line[-1])
            mwu_counter[mwu] = freq

    return mwu_counter