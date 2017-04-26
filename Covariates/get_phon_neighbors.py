"""
adapted from: https://github.com/uconn-maglab/get-neighbors

MIT License

Copyright (c) 2016 Rachael Steiner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from CMUdict.get_cmu_dict import get_cmu_dict


def get_phon_neighbors(target_words):
    """
    The main neighbor-finding method.
    Iterates through the word list, comparing each word in the
    corpus to the current word in length, and passing it to the
    appropriate "checker" function, or moving on if its length
    indicates that it is not a neighbor. If the checker returns
    True, then it appends that word to the current word's "neighbor"
    entry.

    - words = words whose neighbors will be found (target words for regression analysis)
    - neighborhoods_words = words that will be used to find neighbors from (all words in CMU dict)
    """
    cmu_dict = get_cmu_dict()
    neighbors = dict()

    for n_word, word in enumerate(target_words):

        if n_word % 100 == 0:
            print(n_word / len(target_words))

        if word not in cmu_dict:
            continue

        phon_word = cmu_dict[word]
        neighbors[word] = []
        wsplit = phon_word.split('-')
        wlen = len(wsplit)
        for w_neighbor, phon_neighbor in cmu_dict.items():
            qsplit = phon_neighbor.split('-')
            if len(qsplit) == wlen:
                neighbors[word].append(w_neighbor) if check_substitution(wsplit, qsplit) else None
            elif len(qsplit) == wlen+1:
                neighbors[word].append(w_neighbor) if check_addition(wsplit, qsplit) else None
            elif len(qsplit) == wlen-1:
                neighbors[word].append(w_neighbor) if check_deletion(wsplit, qsplit) else None
            else:
                continue

    return neighbors


def check_addition(base, candidate):
    strikes = 0
    for position in range(len(base)):
        while True:
            # If they match, break the while loop and try the next position.
            if base[position] == candidate[position+strikes]:
                break
            # Otherwise, take a strike and continue on that position,
            # as long as it's the first strike. If it's the second strike,
            # then they are not neighbors, so return False.
            else:
                strikes += 1
                if strikes >= 2:
                    return False
    else:
        return True


def check_deletion(base, candidate):
    strikes = 0
    for position in range(len(candidate)):
        while True:
            if base[position+strikes] == candidate[position]:
                break
            else:
                strikes += 1
                if strikes >= 2:
                    return False
    else:
        return True


def check_substitution(base, candidate):
    strikes = 0
    for position in range(len(base)):
        if base[position] == candidate[position]:
            continue
        else:
            strikes += 1
            if strikes >= 2:
                return False
    else:
        return True