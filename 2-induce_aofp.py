import os
import json
from pre_processed_corpora.load_pre_processed_corpus import load_childes_corpus_dict

file_dir = os.path.dirname(os.path.realpath(__file__))

########################################################################################################################


for corpus_name in ['BE', 'NA']:

    # load corpus data
    corpus_dict = load_childes_corpus_dict(corpus_name)

    # order transcripts in corpus by child MLU (from smallest to largest MLU)
    zipped = zip(corpus_dict['child_MLU'], corpus_dict['child_transcripts'])
    ordered = sorted(zipped)

    aofp_dict = dict()

    for mlu, transcript in ordered:
        for utt in transcript:
            for word in utt:
                if word not in aofp_dict:
                    aofp_dict[word] = mlu

    # store AoFP dictionary on disk
    json.dump(aofp_dict, open(os.path.join(file_dir, 'AoFP', '%s.json' % corpus_name), 'w'))
    print('Got AoFP for %s words' % len(aofp_dict))