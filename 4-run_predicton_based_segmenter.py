import os
from optparse import OptionParser
from pre_processed_corpora.load_pre_processed_corpus import load_child_directed_speech
from PredictionBasedSegmenter.PredictionBasedSegmenter import main as run_pbs

file_dir = os.path.dirname(os.path.realpath(__file__))


def corpus_to_txt(corpus_name, temp_path):
    """ Create temporary corpus file, with one utterance per line """
    with open(temp_path, 'w') as temp:
        for u in load_child_directed_speech(corpus_name):
            if len(u) > 0:
                temp.write(' '.join(u) + '\n')
    return temp_path

# set options for PredictionBasedSegmenter
parser = OptionParser()
parser.add_option("-c", "--corpus", dest="corpus",
                  help="the corpus to be segmented, should be tokenized with one sentence per line")
parser.add_option("-v", "--vocab", dest="vocab", help="segment with an existing multiword vocabulary for segmentation")
parser.add_option("-o", "--output", dest="output", help="output a multiword vocabulary with corpus counts")
parser.add_option("-s", "--segment", dest="segment",
                  help="output a corpus segmentation, multiword units are joined by underscore")
parser.add_option("-n", dest="n", default="10", type="int", help="largest n considered for ngrams, default 10")
parser.add_option("-f", "--frequency", dest="frequency", type="int", default=10000000,
                  help="lowest allowed frequency for output vocab, once per FREQUENCY, default 10000000")
parser.add_option("-q", "--quiet", dest="silent", action="store_true", help="don't show progress")
parser.add_option("-l", "--lines", type="int", dest="lines",
                  help="Limit the number of lines (sentences) of the corpus processed, useful for testing")
options, arguments = parser.parse_args()

# set maximum n-gram length to 10
options.n = 10


for corpus_name in ['BE', 'NA']:

    # PredictionBasedSegmenter expects a path to a corpus file with one sentence per line
    # write corpus to a temporary file like that
    temp_path = os.path.join(file_dir, 'PredictionBasedSegmenter', '%s_temp' % corpus_name)
    corpus_path = corpus_to_txt(corpus_name, temp_path)
    options.corpus = corpus_path

    # path to output file with MWUs and frequency counts
    options.output = os.path.join(file_dir, 'PredictionBasedSegmenter', 'MWUs', '%s.txt' % corpus_name)

    run_pbs(options)

    # remove temporary corpus file
    os.remove(temp_path)
