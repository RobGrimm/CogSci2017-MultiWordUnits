import os
import numpy as np
import scipy.stats
from pre_processed_corpora.load_pre_processed_corpus import load_child_directed_speech, load_childes_corpus_dict


file_dir = os.path.dirname(os.path.realpath(__file__))


def get_corpus_statistics():

    ret = dict()
    for corpus_name in ['NA', 'BE']:

        all_tokens = []
        all_types = set()
        utt_lengths = []

        for u in load_child_directed_speech(corpus_name):

            if len(u) == 0:
                continue

            utt_lengths.append(len(u))
            all_tokens += u
            all_types.update(u)

        ret[corpus_name] = (all_tokens, all_types, utt_lengths)

    return ret


def print_corpus_info():

    corpus_statistics_dict = get_corpus_statistics()

    for corpus_name, stats in corpus_statistics_dict.items():

        all_tokens, all_types, utt_lengths = stats

        print(corpus_name)
        print('nr tokens: %s' % len(all_tokens))
        print('nr types: %s' % len(all_types))
        print('nr sentences: %s' % len(utt_lengths))
        print('median utterance length: %s (std: %s)' % (np.median(utt_lengths), scipy.stats.iqr(utt_lengths)))
        print_speaker_info(corpus_name)
        print()


def print_speaker_info(corpus_name):

    d = load_childes_corpus_dict(corpus_name)

    all_adults = []
    for adults in d['adult_names']:
        all_adults += adults

    all_children = []
    for children in d['child_names']:
        all_children += [children]

    ages = d['child_ages']
    ages = [i for i in ages if i and i != 336] # some outlier

    print('nr of adult speakers: %s' % len(set(all_adults)))
    print('nr of child speakers: %s' % len(set(all_children)))
    print('mean child age: %s (SD: %s)' % (np.mean(ages), np.std(ages)))


if __name__ == '__main__':
    print_corpus_info()

