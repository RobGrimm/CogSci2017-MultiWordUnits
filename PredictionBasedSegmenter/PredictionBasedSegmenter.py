##########################################################################
# This program implements the multiword unit segmentation described in
#
# Julian Brooke, Vivian Tsang, Graeme Hirst, and Fraser Shein
# Unsupervised Multiword Segmentation of Large Corpora using
# Prediction-Driven Decomposition of \textit{n}-grams
# Proceedings of COLING 2014.
#
# Run the program without arguments to see a list of options
# This program expects a tokenized corpus with one sentence per line, and
# a space between each word. For a differently formated corpus, change
# read_sentence_from_corpus below
# Note that the once per x tokens threshold should be chosen so that the minimum
# frequency is greater than 1 (and perferably much greater than 1)
##########################################################################

multiword_delimiter = "_" #this is the character used to join multiword segments
                          #in corpus output
import gc
import math
import copy
from pre_processed_corpora.load_pre_processed_corpus import load_child_directed_speech
gc.disable()


def read_sentence_from_corpus(f):
    '''
    The default corpus reader. Change if needed
    '''
    for line in f:
        sentence = line.strip().split()
        for i in range(len(sentence)):
            sentence[i]= sentence[i].lower()
        yield sentence


def is_alpha_plus(word):
    return word.replace("'","").replace("-", "").isalpha()


def add_to_wild_trie(words,start,wild,end,trie):
    curr = trie
    for i in range(start,end):
        if i == wild:
            word = -3
        else:
            word = words[i]
        if word not in curr:
            curr[word] = {}
        curr = curr[word]
    curr[-1] = curr.get(-1,0) + 1

def output_trie(trie,fout,so_far,rev_index):
    for word in trie:
        if word == -1:
            fout.write("%s\t%d\n" % (so_far,trie[-1]))
        else:
            if so_far:
                new_so_far = " ".join([so_far, rev_index[word]])
            else:
                new_so_far = rev_index[word]
            output_trie(trie[word],fout,new_so_far,rev_index)

def trie_match(words,start,end,trie):
    curr = trie
    try:
        for i in range(start,end):
            curr = curr[words[i]]
    except:
        return False
    return -1 in curr

def wild_card_match(words,start,end,wild,trie):
    curr = trie
    try:
        for i in range(start,end):
            if i == wild:
                curr = curr[-3]
            else:
                curr = curr[words[i]]
        return -1 in curr
    except:

        return False

def add_to_trie(words,start,end,trie):
    curr = trie
    for i in range(start,end):
        if words[i] not in curr:
            curr[words[i]] = {}
        curr = curr[words[i]]
    curr[-1] = curr.get(-1,0) + 1

def add_to_trie_all(words,trie,freq):
    curr = trie
    for word in words:
        if word not in curr:
            curr[word] = {}
        curr = curr[word]
    curr[-1] = curr.get(-1,0) + freq


def get_all_trie_entries(trie,so_far):
    for word in trie:
        if word == -1:
            yield so_far,trie[-1]
        else:
            for pair in get_all_trie_entries(trie[word],so_far + [word]):
                yield pair


def move_to_new_trie(source_trie,dest_trie,freq_cutoff):
    for entry,freq in get_all_trie_entries(source_trie,[]):
        if freq >= freq_cutoff:
            curr = dest_trie
            for word in entry:
                if word not in curr:
                    curr[word] = {}
                curr = curr[word]
            curr[-1] = freq

def replace_words_with_index(words,index_dict):
    for i in range(len(words)):
        if words[i] in index_dict:
            words[i] = index_dict[words[i]]
        else:
            words[i] = -2
    return words

def reverse_dict(dictionary):
    return dict((v,k) for k, v in list(dictionary.items()))

def get_ngram_tries(options):
    '''
    This function creates a trie consisting of all the ngrams with n less
    than or equal to largest_n in the corpus named corpus_filename that appear
    at least once per "once_per_cutoff" tokens in the corpus. A separate wild
    card version (with one optional position) is created for n-grams of n >= 3
    '''

    final_trie = {}
    final_wild_card_trie = {}
    index_dict = {}
    token_count = 0
    temp_trie = {}
    temp_wild_card_trie = {}
    corpus_filename = options.corpus
    f = open(corpus_filename)
    sent_count = 0
    if not options.silent:
        print("collecting unigrams")
    for words in read_sentence_from_corpus(f):
        for word in words:
            token_count += 1
            if is_alpha_plus(word):
                temp_trie[word] = temp_trie.get(word,0) + 1
        sent_count += 1
        if sent_count % 100000 == 0 and not options.silent:
            print((str(sent_count) + " sentences done"))
        if options.lines == sent_count:
            break
    f.close()
    token_count = float(token_count)
    freq_cutoff = math.ceil(token_count/options.frequency)
    if not options.silent:
        print(("total tokens in corpus: %d" % token_count))
        print(("token frequency cutoff: %d" % freq_cutoff))

    for word in temp_trie:
        if temp_trie[word] >= freq_cutoff:
            index_dict[word] = len(index_dict)
            final_trie[index_dict[word]] = {-1:temp_trie[word]}
    temp_trie = {}
    if not options.silent:
        print("done collecting unigrams")
    f = open(corpus_filename)
    sent_count = 0
    if not options.silent:
        print("collecting bigrams")
    for sentence in read_sentence_from_corpus(f):
        sentence = replace_words_with_index(sentence,index_dict)
        for j in range(len(sentence) - 1):
            if sentence[j] >=0 and sentence[j+1] >= 0:
                add_to_trie(sentence,j,j+2,temp_trie)
        sent_count += 1
        if sent_count % 100000 == 0 and not options.silent:
            print((str(sent_count) + " sentences done"))
        if options.lines == sent_count:
            break
    f.close()
    move_to_new_trie(temp_trie,final_trie,freq_cutoff)
    wild_index = len(index_dict)
    if not options.silent:
        print("done collecting bigrams")
    for gram in range(3,options.n+1):
        sent_count = 0
        f = open(corpus_filename)
        temp_trie = {}
        if not options.silent:
            print(("collecting %d-grams" % gram))
        for sentence in read_sentence_from_corpus(f):
            sentence = replace_words_with_index(sentence,index_dict)
            last_matched = False
            last_wild_matched = set()
            for i in range(len(sentence) - (gram - 1)):
                if trie_match(sentence,i,i+gram - 1,final_trie):
                    matched = True
                    if matched and last_matched:
                         add_to_trie(sentence,i-1,i+gram-1,temp_trie)
                else:
                    matched = False
                last_matched = matched

                wild_matched = set()

                if gram == 3:
                    if i > 0 and sentence[i - 1] >=0 and sentence[i+1] >=0:
                        add_to_wild_trie(sentence,i-1,i,i+2,temp_wild_card_trie)
                else:

                    for k in range(1,gram - 2):

                        if wild_card_match(sentence,i,i+gram-1,i+k,final_wild_card_trie):
                            wild_matched.add(k)

                            if k+ 1 in last_wild_matched or (k == gram - 3 and trie_match(sentence, i-1,i+k,final_trie)):
                                add_to_wild_trie(sentence,i-1,i + k, i+gram - 1,temp_wild_card_trie)

                    if 1 in last_wild_matched and trie_match(sentence, i+1,i+gram -1,final_trie):
                        add_to_wild_trie(sentence,i-1,i, i+gram-1,temp_wild_card_trie)
                last_wild_matched = wild_matched
            sent_count += 1
            if sent_count % 100000 == 0 and not options.silent:
                print((str(sent_count) + " sentences done"))
            if options.lines == sent_count:
                break
        f.close()
        move_to_new_trie(temp_trie,final_trie,freq_cutoff)
        move_to_new_trie(temp_wild_card_trie,final_wild_card_trie,freq_cutoff)
        if not options.silent:
            print(("done collecting %d-grams" % gram))
    if not options.silent:
        print("done collecting n-grams")
    return final_trie,final_wild_card_trie,index_dict,freq_cutoff,token_count


def get_ngram_spans(words,trie):
    '''
    This function finds maximal matches between ngrams in the trie and
    the words in words. It returns a list of list of spans, where each
    list of lists represents an independent set of spans (no overlaps)
    '''
    ngram_spans = [[]]
    for i in range(len(words)):
        j = i
        curr_level = trie
        local_stop = False
        while not local_stop:
            if j < len(words) and words[j] in curr_level:
                curr_level = curr_level[words[j]]
                j += 1
            else:
                local_stop = True
        length = j - i
        if length > 1:
            if len(ngram_spans[-1]) != 0 and j == ngram_spans[-1][-1][1]:
                pass
            else:
                ngram_spans[-1].append([i,j])
        else:
            if ngram_spans[-1]:
                ngram_spans.append([])
    if not ngram_spans[-1]:
        del ngram_spans[-1]
    return ngram_spans

def get_count_regular_trie(words,start,end,trie):
    curr_level = trie
    for i in range(start,end):
        curr_level = curr_level[words[i]]
    return curr_level[-1]

def get_count_wild_card_trie(words,start,end,wc_index, wild_card_trie):
    curr_level = wild_card_trie
    for i in range(start,end):
        if wc_index == i:
            curr_level = curr_level[-3]
        else:
            curr_level = curr_level[words[i]]
    return curr_level[-1]

def in_span(span,point):
    return span[1] > point >= span[0]

def convert_span(span_str):
    return list(map(int,span_str.split("-")))

def get_len(span_str):
    start,end = span_str.strip().split("-")
    return int(end) - int(start)

def get_ranking_of_predictors(words, index, spans,trie,wild_card_trie,total_token_count):
    '''
    this function uses existing language models stored in trie and wild_card_trie
    to create a ranking of the best spans for the predicting word at index in
    words, limited to the contexts given in spans
    '''

    relevant_spans = []
    for span in spans:
        if in_span(span,index):
           relevant_spans.append(span)
    results = {}
    for span in relevant_spans:
        for i in range(index + 1, span[1] + 1):
            for j in range(index,span[0] - 1,-1):
                if i - j == 1:
                    continue
                rep = str(j) + "-" +str(i)
                if rep in results:
                    continue
                if j == index:
                    results[rep] = float(get_count_regular_trie(words,index,i,trie))/get_count_regular_trie(words,index + 1,i,trie)
                elif i == index + 1:
                    results[rep] = float(get_count_regular_trie(words,j,index + 1,trie))/get_count_regular_trie(words,j, index, trie)
                else:
                    results[rep] = float(get_count_regular_trie(words,j,i,trie))/get_count_wild_card_trie(words,j,i,index,wild_card_trie)
                results[rep] = math.log(results[rep],2)
    results[str(index) + "-" + str(index+1)] = math.log(float(get_count_regular_trie(words,index,index+1,trie))/total_token_count,2)
    to_sort = list(zip(list(results.values()), list(map(get_len,list(results.keys()))), list(map(convert_span,list(results.keys())))))
    to_sort.sort()
    return to_sort

def get_next_breaks(so_far,span_satisfy_dict,max_index):
    i = 0
    while i <= max_index:
        if not span_satisfy_dict[i].intersection(so_far):
            return span_satisfy_dict[i]
        i += 1
    return set()

def has_subset(curr_set,final_set):
    curr_set = set(curr_set)
    for compare_set in final_set:
        if curr_set.issuperset(compare_set):
            return True
    return False

def get_all_minimum_breaks_local(so_far, span_satisfy_dict,max_index):
    '''
    Given a list of spans which need to be broken in span_satisify_dict,
    a list of breaks so_far, and the index of maximum left-most span that
    should be considered in the current context, this function returns
    a list of the minimal breaks that can satisfy the requirements
    '''
    todo_list =  [list(so_far)]
    final_set = []
    while todo_list:
        curr = todo_list.pop(0)
        new_breaks = get_next_breaks(curr,span_satisfy_dict,max_index)
        if not new_breaks:
            if not has_subset(curr,final_set):
                curr.sort()
                final_set.append(curr)
        else:
            for new_break in new_breaks:
                new = curr + [new_break]
                todo_list.append(new)
    return final_set

def get_max_index(spans,curr_index):
    far_left = spans[curr_index +1][1] -1
    new = curr_index + 2
    while new < len(spans) and spans[new][0] <= far_left:
        new += 1
    return new - 2

def compatible_with_breaks(span,breaks):
    for i in range(span[0]+1,span[1]):
        if i in breaks:
            return False
    return True

def span_internal_break(span,point):
    return span[1] > point >= span[0] +1

def break_span(span,point):
    return [span[0],point], [point,span[1]]

def span_in_span(span1,span2):
    return span1[0] >= span2[0] and span1[1] <= span2[1]


def evaluate_breaks(breaks,spans,predictor_dict,max_index,last_break):
    '''
    this function evaluates a set of breaks in the span between last_break
    and the end of the span in spans indexed by max_index, choosing the best
    predicting context that is compatible with the breaks to calculate a log
    probability for the entire span in the context
    '''

    total = 0
    for i in range(last_break,spans[max_index + 1][1]):
        j = len(predictor_dict[i]) - 1
        while not compatible_with_breaks(predictor_dict[i][j][-1], breaks):
            j -= 1
        total += predictor_dict[i][j][0]
    return total

def chunk_sentence_by_ngram_decomposition(words,trie,wild_card_trie,total_token_count):
    '''
    this is the main function for segmenting a sentence (words) based on the
    ngram information in trie and wild_card_trie. The function finds the
    n-gram spans that overlap, identifies the best possible breaks based on
    maximizing the prediction of words in non-broken spans, and then converts
    the orginal (overlapping) ngrams into a list of non-overlaping spans
    which are returned.
    '''

    if len(words) == 0:
        return []

    ngram_spans = get_ngram_spans(words,trie)
    final_chunks = []
    predictor_dict = {}
    for ngram_span_set in ngram_spans:
        span_satisfy_dict = {}
        for i in range(len(ngram_span_set)-1):
            span_satisfy_dict[i] = set(range(ngram_span_set[i+1][0],ngram_span_set[i][1] + 1))
        for span in ngram_span_set:
            for i in range(span[0],span[1]):
                if i not in predictor_dict:
                    predictor_dict[i] = get_ranking_of_predictors(words, i, ngram_span_set,trie,wild_card_trie,total_token_count)

        new_breaks = set()
        for i in range(len(ngram_span_set)-1):
            if new_breaks:
                last_break = max(new_breaks)
            else:
                last_break = ngram_span_set[0][0]
            if last_break in span_satisfy_dict[i]:
                continue
            max_index = get_max_index(ngram_span_set,i)
            all_possible_breaks = get_all_minimum_breaks_local(new_breaks, span_satisfy_dict,max_index)
            best = -9999
            best_breaks = False
            for breaks in all_possible_breaks:
                score = evaluate_breaks(breaks,ngram_span_set,predictor_dict,max_index,last_break)
                if score > best:
                   best = score
                   best_breaks = breaks
            if last_break != ngram_span_set[0][0]:
                old_index = best_breaks.index(last_break)
            else:
                old_index = -1
            new_breaks.add(best_breaks[old_index + 1])
        new_spans = []
        span_set = ngram_span_set
        for new_break in new_breaks:
            for span in span_set:
                if span_internal_break(span,new_break):
                    found_dub1 = False
                    found_dub2 = False
                    span1,span2 = break_span(span,new_break)
                    for i in range(len(span_set)):
                        if span_set[i] != span:
                            if span_in_span(span1, span_set[i]):
                                found_dub1 = True
                            if span_in_span(span2, span_set[i]):
                                found_dub2 = True
                    if not found_dub1 and span1[1] - span1[0] > 1:
                        new_spans.append(span1)
                    if not found_dub2 and span2[1] - span2[0] > 1:
                        new_spans.append(span2)
                else:
                    new_spans.append(span)
            span_set = new_spans
            new_spans = []
        final_chunks.extend(span_set)
    final_chunks.sort()
    return final_chunks

def get_break_subspans(span,words,break_trie):
    current = break_trie
    for i in range(span[0],span[1]):
        try:
            current = current[words[i]]
        except:
            return [span]
    try:
        breakpoint = current[-1]
    except:
        return [span]

    return get_break_subspans([span[0],span[0] + breakpoint],words,break_trie) + get_break_subspans([span[0] + breakpoint,span[1]],words,break_trie)

def make_output(words,spans):
    diff = 0
    for chunk in spans:
        words = words[:chunk[0] - diff] + [multiword_delimiter.join(words[chunk[0] -diff:chunk[1] - diff])] + words[chunk[1]-diff:]
        diff += chunk[1] - chunk[0] - 1
    return " ".join(words) + "\n"

def segment_corpus_by_decomposition(index_dict, ngram_trie,wild_card_ngram_trie,total_token_count,options,break_trie={}):
    '''
    this function segments an entire corpus using the information in the tries.
    If output is given, it will output the resulting segmentation with
    underscores linking multiword chunks. If break_trie is given, any chunked
    span which matches a entry in break_trie will be further decomposed
    recursively until there are no spans which match an entry. The function
    returns a trie representing the counts of each segment in the corpus
    '''

    if options.segment and (options.vocab or break_trie):
        fout = open(options.segment,"w")
    if not options.silent:
        if break_trie:
            print("resegmenting corpus with new breaks")
        elif not options.vocab:
            print("starting initial corpus segmentation")
        else:
            print("segmenting corpus with exisiting multiword vocabulary")

    count_trie = {}
    f = open(options.corpus)
    sent_count = 0
    for sentence in read_sentence_from_corpus(f):
        org_sentence = copy.copy(sentence)
        sentence = replace_words_with_index(sentence,index_dict)
        chunked_spans = chunk_sentence_by_ngram_decomposition(sentence,ngram_trie,wild_card_ngram_trie,total_token_count)
        if break_trie:
            new_spans = []
            for span in chunked_spans:
                new_spans.extend(get_break_subspans(span,sentence,break_trie))
            chunked_spans = new_spans
        if options.segment and (options.vocab or break_trie):
            fout.write(make_output(org_sentence,chunked_spans))
        taken_indicies = set()
        for span in chunked_spans:
            add_to_trie(sentence,span[0],span[1],count_trie)
            taken_indicies.update(list(range(span[0],span[1])))
        for i in range(len(sentence)):
            if i not in taken_indicies and sentence[i] != -2:
                add_to_trie(sentence,i,i+1,count_trie)
        sent_count += 1
        if not options.silent and sent_count % 100000 == 0:
            print((str(sent_count) + " sentences done"))
        if options.lines == sent_count:
            break
    f.close()
    if not options.silent:
        print("segmentation complete")
    return count_trie

def get_best_over_range(start,end,predictor_dict):
    total = 0
    for i in range(start,end):
        total += predictor_dict[i][-1][0]
    return total

def get_best_over_range_limit(start,end,predictor_dict):
    total = 0
    for i in range(start,end):
        j = -1
        while True:
            #print "here"
            if predictor_dict[i][j][-1][0] >= start and predictor_dict[i][j][-1][1] <= end:
                total += predictor_dict[i][j][0]
                break
            j -= 1
    return total


def decompose_vocabulary(counts_trie,ngram_trie,wild_card_trie,total_token_count,freq_cutoff,options):
    '''
    This function takes an raw multiword vocabulary based on an initial segmentation
    and returns a new one, spliting all multiword units which are less than the frequency
    cutoff or (for length > 2) where the decrease in token count is not offset by an
    increase in predictiveness. The best split point is also based on maximum
    predictiveness. The algorithm starts with the largest n-grams, and works
    towards the smallest, so the parts from each split are each considered in a
    later iteration.
    '''
    if not options.silent:
        print("decomposing vocabulary")
    break_trie = {}
    for k in range(options.n-1,0,-1):
        if not options.silent:
            print(("%d-grams" % (k + 1)))
        new_trie = {}
        not_broke_count = 0
        broke_count = 0
        for words,count in get_all_trie_entries(counts_trie,[]):
            if len(words) == k + 1:
                if k == 1:
                    best_break = 1
                else:
                    best = -9999
                    best_break = False
                    predictor_dict = {}
                    for i in range(len(words)):
                        predictor_dict[i] = get_ranking_of_predictors(words, i, [[0,len(words)]],ngram_trie,wild_card_trie,total_token_count)
                    for i in range(1, len(words)):
                        score = evaluate_breaks([i],[[0,len(words)]],predictor_dict,-1,0)
                    if score > best:
                        best = score
                        best_break = i
                sub_ngram1 = words[:best_break]
                try:
                    sub_ngram1_count = get_count_regular_trie(sub_ngram1,0,len(sub_ngram1),counts_trie)
                except:
                    sub_ngram1_count = 0
                sub_ngram2 = words[best_break:]
                try:
                    sub_ngram2_count = get_count_regular_trie(sub_ngram2,0,len(sub_ngram2),counts_trie)
                except:
                    sub_ngram2_count = 0
                if count >= freq_cutoff and (k==1 or (sub_ngram1_count ==0 or math.log(sub_ngram1_count,2) - math.log(count,2) < get_best_over_range(0,best_break,predictor_dict) - get_best_over_range_limit(0,best_break,predictor_dict)) and (sub_ngram2_count == 0 or  math.log(sub_ngram2_count,2) - math.log(count,2) < get_best_over_range(best_break,len(words),predictor_dict) - get_best_over_range_limit(best_break,len(words),predictor_dict))):
                    add_to_trie_all(words,new_trie,count)
                    not_broke_count += 1
                else:
                    broke_count +=1
                    add_to_trie_all(sub_ngram1,new_trie,count)
                    add_to_trie_all(sub_ngram2,new_trie,count)
                    add_to_trie_all(words,break_trie,best_break)
            else:
                add_to_trie_all(words,new_trie,count)
        if not options.silent:
            print("removed")
            print(broke_count)
            print("preserved")
            print(not_broke_count)
        counts_trie = new_trie
    vocab_count = 0
    for words,count in get_all_trie_entries(counts_trie,[]):
        vocab_count += 1
    if not options.silent:
        print("vocabulary decomposition complete")
    return counts_trie,break_trie

def load_multiword_vocab(options):
    '''
    This function takes a multiword vocabulary (with counts) and creates
    ngram tries (regular and with one "wild card") from it, as well as an
    index dictionary
    '''
    if not options.silent:
        print("loading vocab")
    f = open(options.vocab)
    index_dict = {}
    trie = {}
    wild_card_trie = {}
    total_tokens = 0
    for line in f:
        ngram, count = line.strip().split("\t")
        org_ngram = ngram
        count = int(count)
        total_tokens += count
        ngram = ngram.split()
        for i in range(len(ngram)):
            if ngram[i] not in index_dict:
                index_dict[ngram[i]] = len(index_dict)
            ngram[i] = index_dict[ngram[i]]
        for i in range(len(ngram)):
            for j in range(i+ 1,len(ngram)+1):
                curr_level = trie
                for word in ngram[i:j]:
                    if not word in curr_level:
                        curr_level[word] = {}
                    curr_level = curr_level[word]
                curr_level[-1] = curr_level.get(-1,0) + count
                if j - i >= 3:
                    for k in range(i+1,j-1):
                        curr_level = wild_card_trie
                        for p in range(i,j):
                            if p == k:
                                word = -3
                            else:
                                word = ngram[p]
                            if not word in curr_level:
                                curr_level[word] = {}
                            curr_level = curr_level[word]
                        curr_level[-1] = curr_level.get(-1,0) + count
    return trie,wild_card_trie,index_dict,total_tokens


def main(options):
    reg_trie,wild_trie,index_dict,freq_cutoff,total_token_count = get_ngram_tries(options) #"unchunked_test.txt")
    rev_index_dict = reverse_dict(index_dict)
    counts_trie = segment_corpus_by_decomposition(index_dict, reg_trie,wild_trie,total_token_count,options)
    counts_trie,break_trie = decompose_vocabulary(counts_trie,reg_trie,wild_trie,total_token_count,freq_cutoff,options)
    if options.output:
        fout = open(options.output,"w")
        output_trie(counts_trie,fout,[],rev_index_dict)
    if options.segment:
        segment_corpus_by_decomposition(index_dict, reg_trie,wild_trie,total_token_count,options,break_trie=break_trie)