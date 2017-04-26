import os


file_dir = os.path.dirname(os.path.realpath(__file__))


def get_cmu_dict():
    """
    Return dictionary mapping orthographic word forms to sallaybified phonemic transcriptions
    Use the CMU pronounciation dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
    """
    cmu_dict = dict()
    with open(file_dir + '/cmu_dict.txt', 'r') as file_:
            for idx, line in enumerate(file_):
                word_phonemes = line.split()
                word = word_phonemes[0].lower()
                phonemes = word_phonemes[1:]
                cmu_dict[word] = ''.join(phonemes)
    return cmu_dict