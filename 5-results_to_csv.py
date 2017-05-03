import os
import csv
from AoFP.load_aofp import load_age_of_first_production
from pre_processed_corpora.load_freq_dict import load_freq_dict
from ChunkBasedLearner.load_mwus import load_mwus as load_cbl_mwus
from PredictionBasedSegmenter.load_mwus import load_mwus as load_pbs_mwus
from Covariates.get_covariates import get_concretness_values, get_nsyllables, get_phon_neighbor_dict


file_dir = os.path.dirname(os.path.realpath(__file__))


def get_number_mwus_per_word(mwu_counter, min_freq=10):
    """
    :param mwu_counter: dictionary mapping MWUs to their frequency counts
    :return: dictionary mapping words to the number of MWUs within which they occur
    """
    ret = dict()
    for mwu, freq in mwu_counter.items():

        if freq < min_freq:
            continue

        for w in mwu:
            if w not in ret:
                ret[w] = 0
            ret[w] += 1
    return ret


def get_number_contextws_by_word(mwu_counter, min_freq=10):
    """
    :param mwu_counter: dictionary mapping MWUs to their frequency counts
    :return: dictionary mapping each target word to the number of distinct context words that appear in the
             MWUs which contain the target words
    """
    ret = dict()
    for mwu, freq in mwu_counter.items():

        if freq < min_freq:
            continue

        for w in mwu:
            if w not in ret:
                ret[w] = 0
            ret[w] += len(mwu) - 1
    return ret


########################################################################################################################

# get results for each corpus
for corpus_name in ['BE', 'NA']:

    # use AoFP from other corpus
    if corpus_name == 'BE':
        aofp_dict = load_age_of_first_production('NA')
    elif corpus_name == 'NA':
        aofp_dict = load_age_of_first_production('BE')
    else:
        raise Exception('Corpus must be one of: BE, NA')

    # dictionaries mapping words to frequency and phonological neighbors
    freq_dict = load_freq_dict(corpus_name)
    neighbor_dict = get_phon_neighbor_dict()

    # only consider target words if we have freuency counts and phonological neighbors
    target_words = [w for w in aofp_dict if w in freq_dict and w in neighbor_dict]

    # lists containing covariates
    phon_neighbors = [len(neighbor_dict[w]) for w in target_words]
    concreteness = get_concretness_values(target_words)
    freq_per_word = [freq_dict[w] for w in target_words]
    nsyls = get_nsyllables(target_words)

    # list with AoFP values
    aofp = [aofp_dict[w] for w in target_words]

    # get number of MWUs for each tareget word (for the Prediction Based Segmenter)
    pbs_mwu_counter = load_pbs_mwus(corpus_name)
    nr_pbs_mwus_by_word = get_number_mwus_per_word(pbs_mwu_counter)
    pbs_mwus = [nr_pbs_mwus_by_word[w] if w in nr_pbs_mwus_by_word else 0 for w in target_words]

    # get number of context words contained in MWUs for each target word
    pbs_contextws_by_word = get_number_contextws_by_word(pbs_mwu_counter)
    pbs_ctxtws = [pbs_contextws_by_word[w] if w in pbs_contextws_by_word else 0 for w in target_words]

    # perform same steps for the Chunk Based Learner
    cbl_mwu_counter = load_cbl_mwus(corpus_name)
    nr_cbl_mwus_by_word = get_number_mwus_per_word(cbl_mwu_counter)
    cbl_mwus = [nr_cbl_mwus_by_word[w] if w in nr_cbl_mwus_by_word else 0 for w in target_words]

    cbl_contextws_by_word = get_number_contextws_by_word(cbl_mwu_counter)
    cbl_ctxtws = [cbl_contextws_by_word[w] if w in cbl_contextws_by_word else 0 for w in target_words]

    # write results to CSV
    with open(os.path.join(file_dir, '%s.csv' % corpus_name), 'w') as file_:

        writer = csv.writer(file_, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        measures = ['word', 'AoFP', 'freq', 'concreteness', 'nsyl', 'phon_n',
                    'PBS', 'PBS_ctxtws',
                    'CBL', 'CBL_ctxtws']

        writer.writerow(measures)

        for row in zip(*[target_words, aofp, freq_per_word, concreteness, nsyls, phon_neighbors,
                         pbs_mwus, pbs_ctxtws,
                         cbl_mwus, cbl_ctxtws]):

            writer.writerow(row)