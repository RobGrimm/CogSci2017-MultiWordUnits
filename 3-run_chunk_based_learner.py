import os
import pickle
from pre_processed_corpora.load_pre_processed_corpus import load_child_directed_speech
from ChunkBasedLearner.ChunkBasedLearner import ChunkBasedLearner


file_dir = os.path.dirname(os.path.realpath(__file__))


########################################################################################################################


for corpus_name in ['BE', 'NA']:

    utterances = load_child_directed_speech(corpus_name)

    cbl = ChunkBasedLearner(random_btp=False)
    mwu_counter = cbl.get_mwus(utterances)
    with open(os.path.join(file_dir, 'ChunkBasedLearner', 'MWUs', '%s.pickle' % corpus_name), 'wb') as f:
        pickle.dump(mwu_counter, f, protocol=pickle.HIGHEST_PROTOCOL)