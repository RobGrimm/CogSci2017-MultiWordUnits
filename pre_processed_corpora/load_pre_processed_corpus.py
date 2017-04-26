import os
import json


file_dir = os.path.dirname(os.path.realpath(__file__))


def load_childes_corpus_dict(corpus_name):
    assert corpus_name in ['BE', 'NA']
    corpus_dict = json.load(open(os.path.join(file_dir, '%s.json' % corpus_name), 'r'))
    return corpus_dict


def load_child_directed_speech(corpus_name):

    # CDS corpus is stored in dictionary format and needs to be re-factored before we can return it
    corpus_dict = load_childes_corpus_dict(corpus_name)

    # 1) order transcripts by child MLU
    zipped = zip(corpus_dict['child_MLU'], corpus_dict['adult_transcripts'])
    ordered = sorted(zipped)
    transcripts = [t for mlu, t in ordered]

    # 2) convert list of transcripts into flat list of utterances
    utterances = []
    for u in transcripts:
        utterances.extend(u)

    return utterances