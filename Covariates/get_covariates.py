import os
import pickle
from nltk.stem import WordNetLemmatizer
from CMUdict.get_cmu_dict import get_cmu_dict
from Covariates.get_phon_neighbors import get_phon_neighbors

file_dir = os.path.dirname(os.path.realpath(__file__))


def get_concreteness_dict():
    """
    Return dictionary mapping lemmas to concreteness ratings
    Concreteness ratings are from:
        Brysbaert, M., Warriner, A.B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known
        English word lemmas. Behavior Research Methods, 46, 904-911. (http://crr.ugent.be/archives/1330)
    """
    ret = dict()

    with open('/'.join([file_dir, 'concreteness.txt']), 'r') as file_:

        for idx, line in enumerate(file_):

            # skip column names
            if idx == 0:
                continue

            line = [i.strip() for i in line.split()]

            if line == ['']:
                continue

            # more than two word entries
            # these are multi-word expressions (e.g. 'boarding school'), not single words
            if len(line) != 9:
                continue

            word, bigram, conc_mean, conc_sd, unk, total, perc_kmown, subtlex_pos, dom_pos = line
            ret[word] = float(conc_mean)

    return ret


def get_concretness_values(target_words):
    """
    Given a list of target words, return a list of concreteness ratings corresponding to target words.
    If there is a concreteness rating for the target word, use that. Else, see if there is a concreteness rating
    for the lemma of the target word and use that instead.
    """
    ret = []

    lem = WordNetLemmatizer()
    conc_dict = get_concreteness_dict()

    for w in target_words:

        # use target word's concreteness rating if there is one
        if w in conc_dict:
            ret.append(conc_dict[w])
        # failing that, lemmatize the target word and use the lemma's concreteness rating
        else:
            lemma = lem.lemmatize(w)
            if lemma in conc_dict:
                ret.append(conc_dict[lemma])
            else:
                ret.append(None)

    return ret


def get_nsyllables(target_words):
    """ Given a list of target words, return a list of syllable lengths corresponding to the target words """
    cmu_dict = get_cmu_dict()
    ret = [len(cmu_dict[w].split('-')) if w in cmu_dict else 0 for w in target_words]
    return ret


def get_phon_neighbor_dict():
    """
    Return a dictionary mapping all words in the CMU pronounciation dictionary to their phonological neighbors
    (phonological neighbors are all words whose phonemic form can be obtained by deleting, adding, or substituting
    a single phoneme in the target word).
    """
    try:
        # return the phonological neighbors dictionary if it already exists
        neighbor_dict = pickle.load(open(os.path.join(file_dir, 'phon_neighbor_dict.pickle'), 'rb'))
        return neighbor_dict
    except FileNotFoundError:
        # else, create it from scratch, pickle it, and then return it
        print('Creating dictionary with phonological neighbors')

        cmu_dict = get_cmu_dict()
        target_words = list(cmu_dict.keys())

        neighbor_dict = get_phon_neighbors(target_words)

        with open(os.path.join(file_dir, 'phon_neighbor_dict.pickle'), 'wb') as f:
            pickle.dump(neighbor_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        return neighbor_dict