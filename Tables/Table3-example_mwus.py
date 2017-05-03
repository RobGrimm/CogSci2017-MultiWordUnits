from ChunkBasedLearner.load_mwus import load_mwus as load_cbl_mwus
from PredictionBasedSegmenter.load_mwus import load_mwus as load_pbs_mwus


def get_first_x_mwus_with_word(freqs_to_mwus, target_word, nr_examples):

    # frequencies in decreasing order
    freqs = sorted(freqs_to_mwus.keys(), reverse=True)

    example_mwus = []
    for f in freqs:
        # sorted is for replicability, as order can vary on different runs
        mwus = sorted(freqs_to_mwus[f])

        for mwu in mwus:
            words = set(mwu)

            if target_word in words:
                example_mwus.append((mwu, f))

        if len(example_mwus) >= nr_examples:
            break

    return example_mwus


def map_frequncies_to_mwus(mwu_freq_dict):
    freqs_to_mwus = dict()
    for mwu, freq in mwu_freq_dict.items():
        if freq in freqs_to_mwus:
            freqs_to_mwus[freq].append(mwu)
        else:
            freqs_to_mwus[freq] = [mwu]
    return freqs_to_mwus


def print_example_mwus(mwu_type, corpus_name, nr_examples, target_word):

    if mwu_type == 'CBL':
        n_gram_counter = load_cbl_mwus(corpus_name)
    elif mwu_type == 'PBS':
        n_gram_counter = load_pbs_mwus(corpus_name)
    else:
        raise Exception('unknown muw type: %s' % mwu_type)

    freqs_to_mwus = map_frequncies_to_mwus(n_gram_counter)
    example_mwus = get_first_x_mwus_with_word(freqs_to_mwus, target_word, nr_examples)

    for mwu, freq in example_mwus:
        print(' '.join(mwu), freq)


if __name__ == '__main__':
    #print_example_mwus(mwu_type='PBS', corpus_name='NA', nr_examples=5, target_word='girl')
    #print_example_mwus(mwu_type='CBL', corpus_name='NA', nr_examples=5, target_word='girl')
    #print_example_mwus(mwu_type='PBS', corpus_name='NA', nr_examples=5, target_word='sit')
    print_example_mwus(mwu_type='CBL', corpus_name='NA', nr_examples=5, target_word='sit')