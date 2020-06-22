# Importing libraries
import os
import time

os.path.dirname(os.path.realpath('__file__'))

import nltk
import numpy as np
import pandas as pd
import random
import re
from sklearn.model_selection import train_test_split


# reading the Treebank tagged sentences
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))

# Splitting the data into Train and Validation set   #test_set = validation
train, validation = train_test_split(nltk_data, train_size=0.95, test_size=0.05, random_state=101)

# create list of train and test tagged words
train_tagged_words = [tup for sent in train for tup in sent]     # gets all tagged words together out of sentence
print('total word tags in training set : ', len(train_tagged_words))   # 95547


""" -------------------------------------------------------
The NLTK contains 12 Coarse Tags as Following : 

VERB - verbs (all tenses and modes)
NOUN - nouns (common and proper)
PRON - pronouns
ADJ - adjectives
ADV - adverbs
ADP - adpositions (prepositions and postpositions)
CONJ - conjunctions
DET - determiners
NUM - cardinal numbers
PRT - particles or other function words
X - other: foreign words, typos, abbreviations
. - punctuation

-------------------------------------------------------- """
tags = {tag for word, tag in train_tagged_words}    # create a SET of Tags

# A set containing the Words in Training Set
train_vocab = {word for word, tag in train_tagged_words}  # create a SET of Words  -- len 12100


# ----------------------------------- VITERBI ALGORITHM ---------------------------
# compute Emission Probability
def word_given_tag(word, tag, train_bag):  # train_bag=train_tagged_words
    """
    Emission probability of Word(w) and Tag(t) is,
    Number of times word (w) has been tagged with (t) / numb of times t Appear

    :param word: the Word (W)
    :param tag: The  Tag (t) for word (W)
    :param train_bag: corpus of word and their tags
    :return: freq of (w|t), freg of tag (t)
    """
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    count_tag = len(tag_list)  # total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    # now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
    return count_w_given_tag, count_tag


# compute  Transition Probability
def t2_given_t1(t2, t1, train_bag):
    """
    Prob of trainsition of tag(t1) to tag(t2) = no of times (t2|t1)/ num of (t1) appears
    :param t2: tag
    :param t1: tag
    :param train_bag: Reference corpus
    :return: count of (t2|t1), count of t1
    """
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t == t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index] == t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return count_t2_t1, count_t1


# Vanilla Viterbi Algorithm
def Viterbi(words, train_bag):
    state = []
    T = list(set([pair[1] for pair in train_bag]))  # gives list of unique Tags
    for key, word in enumerate(words):
        # initialise list of probability column for a given observation
        p = []
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]      # getting transition prob from Pandas DF created
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            cnt_wrd_tag = word_given_tag(word=words[key],
                                         tag=tag,
                                         train_bag=train_tagged_words)[0]
            cnt_tag = word_given_tag(word=words[key],
                                     tag=tag,
                                     train_bag=train_tagged_words)[1]
            # Emission Probability Calculation
            emission_p = cnt_wrd_tag/cnt_tag
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]
        state.append(state_max)
    return list(zip(words, state))


# Viterbi Algorithm with Rule based
def Viterbi_Rule_Based(words, unknown_word_set, train_bag):
    state = []
    T = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):
        #print(key, word)
        if re.search(r'^-?[0-9]+(.[0-9]+)?$', word):
            if word == '0':
                state_max = 'X'
            else:
                state_max = 'NUM'
        elif re.search('ing',word) or re.search('ed', word):
            state_max = 'VERB'
        elif re.search(r'\*[A-Za-z]*\*-[0-9]*', word):
            state_max = 'X'
        elif re.search(r'-[A-Za-z]*-', word):
            state_max = '.'
        elif word in unknown_word_set:  # tagging unknown words to the Most occured Tag
            state_max = 'NOUN'

        # initialise list of probability column for a given observation
        else:
            p = []
            for tag in T:
                if key == 0:
                    transition_p = tags_df.loc['.', tag]
                else:
                    transition_p = tags_df.loc[state[-1], tag]

                # compute emission and state probabilities
                cnt_wrd_tag = word_given_tag(word=words[key],
                                             tag=tag,
                                             train_bag=train_tagged_words)[0]
                cnt_tag = word_given_tag(word=words[key],
                                         tag=tag,
                                         train_bag=train_tagged_words)[1]
                # Emission Probability Calculation
                emission_p = cnt_wrd_tag/cnt_tag
                state_probability = emission_p * transition_p
                p.append(state_probability)

            pmax = max(p)
            # getting state for which probability is maximum
            state_max = T[p.index(pmax)]
        state.append(state_max)
    return list(zip(words, state))


# Viterbi algiorithm with Rule based +  Probabilistic approch
def Viterbi_rule_prob(words, unknown_list, train_bag):
    state = []
    T = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):
        #print(key, word)
        if re.search(r'^-?[0-9]+(.[0-9]+)?$', word):
            if word == '0':
                state_max = 'X'
            else:
                state_max = 'NUM'
        elif re.search('ing',word) or re.search('ed', word):
            state_max = 'VERB'
        elif re.search(r'\*[A-Za-z]*\*-[0-9]*', word):
            state_max = 'X'
        elif re.search(r'-[A-Za-z]*-', word):
            state_max = '.'
        elif word in unknown_list:
            print(word)
            p=[]
            for tag in T:
                transition_p = tags_df.loc[state[-1], tag]
                state_probability = transition_p
                p.append(state_probability)
            pmax = max(p)
            state_max = T[p.index(pmax)]
        # initialise list of probability column for a given observation
        else:
            p = []
            for tag in T:
                if key == 0:
                    transition_p = tags_df.loc['.', tag]
                else:
                    transition_p = tags_df.loc[state[-1], tag]

                # compute emission and state probabilities
                cnt_wrd_tag = word_given_tag(word=words[key],
                                             tag=tag,
                                             train_bag=train_tagged_words)[0]
                cnt_tag = word_given_tag(word=words[key],
                                         tag=tag,
                                         train_bag=train_tagged_words)[1]

                # Emission Probability Calculation
                emission_p = cnt_wrd_tag/cnt_tag
                state_probability = emission_p * transition_p
                p.append(state_probability)

            pmax = max(p)
            # getting state for which probability is maximum
            state_max = T[p.index(pmax)]
        state.append(state_max)
    return list(zip(words, state))


"""
# creating t * t transition matrix of tags, t= no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)
"""
tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)):
        tags_matrix[i, j] = t2_given_t1(t2, t1,
                                        train_bag=train_tagged_words)[0]/t2_given_t1(t2, t1,
                                                                                     train_bag=train_tagged_words)[1]
# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
del tags_matrix

"""
# The best method to Test is to run the algorithms on a test Sample data
# Here we take a sample of 10 Sentences to run our Algos

random.seed(1234)      # define a random seed to get same sentences when run multiple times
rndom = [random.randint(1,len(validation)) for x in range(5)] # choose random 10 numbers
test_run = [validation[i] for i in rndom]  # SENTENCE
test_run_base = [tup for sent in test_run for tup in sent]  # list of word and tag
test_untagged_words = [tup[0] for sent in test_run for tup in sent]  # only WORDS

"""

# Running our code on complete Test/ Validation Data

# Validation Data Preprocessing
validation_tagged_words = [tup for sent in validation for tup in sent]  # list of word and tag
print('validation_tagged_words :', validation_tagged_words[:5])
validation_vocab = [tup[0] for sent in validation for tup in sent]  # only WORDS
print('validation_vocab :',validation_vocab[:5])

# ------------------------ IMPLEMENTING THE VANILLA VITERBI ALGORITHM ----------------------
start = time.time()
tagged_seq_vanilla = Viterbi(words=validation_vocab,
                             train_bag=train_tagged_words)  # Viterbi on sample Test Data
end = time.time()
difference = end-start

correct_tags_vanilla = [i for i, j in zip(tagged_seq_vanilla, validation_tagged_words) if i == j]
accuracy_vanilla = len(correct_tags_vanilla)/len(tagged_seq_vanilla)

print("Time taken in Minutes: ", difference/60)
print('Total Correct Tags :', len(correct_tags_vanilla))
print('Vanilla Viterbi Algorithm Accuracy: ',accuracy_vanilla*100)

# ---------------------------------------UNKNOWN WORDS ---------------------------------------

"""
Now we come to our Problem Statement :
Improving the Accuracy of Vanilla Viterbi Algorithm on Unknown wrods in our Validation Set

"""
unknown_vocab = list(set(validation_vocab) - set(train_vocab))
print(len(unknown_vocab))

# Finding Most Frequent Tag for unknown Words
unknown_word_tag_freq = dict([(key, 0) for key in tags])   # Dictionary Creation

for ele in validation_tagged_words:
    if ele[0] in unknown_vocab:
        unknown_word_tag_freq[ele[1]] = unknown_word_tag_freq.get(ele[1],0)+1

print(unknown_word_tag_freq)

"""
Note :  From above we see that Most unknown words in the vocablary belongs to the Tag "NOUN" 
        and Assuming a Few Numerical Format Data belong to "NUM".
"""

# -------------------------------------MODIFIED VITERBI ALGORITHM -1 --------------------------

"""
Considering the Observation from Above we will try to Improve the Vanilla Viterbi Algorithm using 
    morphological cues.
    1. Identify Numbers using re.search and tag them as NUM
    2. Assign all Unknown words the most Frequent Tags ie NOUN in our case
    
The Function for this Improved Viterbi is defined above , we will be simply calling it below

"""
start = time.time()
tagged_seq_rule = Viterbi_Rule_Based(words=validation_vocab,
                                     unknown_word_set=unknown_vocab,
                                     train_bag=train_tagged_words)  # Viterbi on sample Test Data
end = time.time()
difference = end-start

correct_tags_rule = [i for i, j in zip(tagged_seq_rule, validation_tagged_words) if i == j]
accuracy_rule_based = len(correct_tags_rule)/len(tagged_seq_rule)

print("Time taken in Minutes: ", difference/60)
print('Total Correct Tags :', len(correct_tags_rule))
print('Rule based Viterbi Algorithm Accuracy: ',accuracy_rule_based*100)

# -------------------------------------MODIFIED VITERBI ALGORITHM -2 --------------------------

"""
In this Modified version we try to incorporate the probabilistic approch to unknown vacablary set
    1. We use Rule based Approch to Tag NUM and X
    2. We use a probabilistic Rule that: 
        for words not in the Training Corpus, their State Probability = Transition Probability
    
The Function for this Improved Viterbi is defined above , we will be simply calling it below

"""

start = time.time()
tagged_seq_prob = Viterbi_rule_prob(validation_vocab, unknown_vocab)  # Viterbi on sample Test Data
end = time.time()
difference = end-start
correct_tags_prob = [i for i, j in zip(tagged_seq_prob, validation_tagged_words) if i == j]
accuracy_probabilistic = len(correct_tags_prob)/len(tagged_seq_prob)

print("Time taken in Minutes: ", difference/60)
print('Total Correct Tags :', len(correct_tags_prob))
print('Probality Viterbi Algorithm Accuracy: ',accuracy_probabilistic*100)


# -------------- Compare the tagging accuracies of the modifications with the vanilla Viterbi algorithm--------

print('The Vanilla Viterbi\n')
print('Accuracy of Vanilla Viterbi Algorithm : ',accuracy_vanilla*100)

print('Modified Viterbi Algorithm -2 \n '
      'uses a Probabilistic rule for unknown words that State Probality of Unknown words  =  Transition Probablity\n')
print('Accuracy of Probality based Viterbi Algorithm : ',accuracy_probabilistic*100)

print('Modified Viterbi Algorithm - 1 \n '
      'uses a Rule based approch for unknown words ie. tagging unknown words with Most occured POS Tag\n')
print('Accuracy of Probality based Viterbi Algorithm : ',accuracy_rule_based*100)


# ---------------------- RESULT of IMPROVEMENTS MADE ---------------------------
"""
List down cases which were incorrectly tagged by original POS tagger and 
got corrected by your modifications
"""
# incorrect tagging by Vanilla Viterbi
incorrect_tags_vanilla = [j for i, j in zip(tagged_seq_vanilla, validation_tagged_words) if i != j]

# Correction in tagging made my RULE based Viterbi (ALGO-1)
corrected_by_Algo_rule = []
for ele in incorrect_tags_vanilla:
    if ele in correct_tags_rule:
        corrected_by_Algo_rule.append(ele[0])
print('The Following Tags got corrected in the Rule based modified Viterbi Algorithm\n')
print(corrected_by_Algo_rule)

# Correction in tagging made my PROB based Viterbi (ALGO-2)
corrected_by_Algo_prob = []
for ele in incorrect_tags_vanilla:
    if ele in correct_tags_prob:
        corrected_by_Algo_prob.append(ele[0])
print('The Following Tags got corrected in the Rule based modified Viterbi Algorithm\n')
print(corrected_by_Algo_prob)






