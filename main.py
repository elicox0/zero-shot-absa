#!/usr/bin/env python
# main.py

import sys
import re
import numpy as np
from typing import Generator

import transformers
import lda
from lda import json_to_text_it, text_it


PREFIX = '/home/eli/Code/absalute-zero'
PATH_TO_DATA   = f'{PREFIX}/foursquare.txt' 
PATH_TO_TOPICS = f'{PREFIX}/topics.txt'
#NUM_TOPICS = 10 # 100 may be suitable for generic corpus of reviews; 20 or fewer for single business

class Topic(object):
    """
    Class representing a topic in the corpus.
    Initialized with a name and list of words.

    Attributes
    ----------
    name <str> : The name of the topic. Can be assigned manually or with a zero-shot classifier.
    words <list[str]> : The list of words pertaining to the topic.

    Methods
    -------
    set_sentences: returns an iterator over all pertinent sentences
    get_avg_sentiment: returns average sentiment for the topic
    """
    def __init__(self, lda_out : list[tuple]) -> None:
        self.words = [t[0] for t in lda_out] # t is tuple of (word, weight)
        self.weights = [t[1] for t in lda_out]

    def set_sentences(self, corpus : Generator[str,None,None]) -> None:
        """
        Sets `self.sentences` to a generator of all sentences in the corpus that 
        contain at least one  word in the list of words corresponding to the topic.
        """
        punc_re = re.compile(r'[\.?!]+')
        self.sentences = []
        for i,doc in enumerate(corpus):
            if i == 0:
                print(doc)
                print()
                print(punc_re.split(doc))
            for seq in punc_re.split(doc):
                if any((re.search(word,seq) for word in self.words)):
                    self.sentences.append(seq)
#        self.sentences = (seq for seq in (punc_re.split(doc) 
#                    for doc in corpus) 
#                    if any((re.search(word, seq) for word in self.words)))

    def set_name(self, clf : transformers.pipeline, labels : list[str]) -> None:
        """
        Sets `self.name` to the best label from the list provided as determined 
        by a zero-shot classifier. `self.sentences` must be set already.
        """
        result = clf(" ".join(self.words), labels)
        self.name = result['labels'][np.argmax(result['scores'])]
        labels.remove(self.name) # if we want to prevent duplicate labels

    def get_avg_sentiment(self) -> float:
        """
        Returns the average sentiment over the object's sentences.
        """
        clf = transformers.pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        return np.mean(np.array([clf(s) for s in self.sentences]))

def pipeline(doc : str, lda : lda.gensim.models.ldamulticore.LdaMulticore) -> dict:
    """
    Returns a dictionary of (topic : avg_sentiment) pairs for doc.
    """
    pass

candidate_labels = sys.argv[1:]
dtype = PATH_TO_DATA.split('.')[-1]
clf = transformers.pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
score_dict = {}
corpus_dict = lda.create_dictionary(PATH_TO_DATA, dtype=dtype)
try:
    ldamodel = lda.get_ldamodel(len(candidate_labels), PATH_TO_DATA, corpus_dict, dtype=dtype)
except ZeroDivisionError as e:
    raise e("Please provide candidate topic names as system arguments.")
topics = [t[1] for t in ldamodel.show_topics(num_topics=-1, formatted=False)]

for topic in topics:
    with open(PATH_TO_DATA, mode=lda.DTYPE_MAP[dtype][1]) as handle:
        t = Topic(topic)
        t.set_sentences(lda.DTYPE_MAP[dtype][0](handle))
        t.set_name(clf, candidate_labels)
        score_dict[t.name] = t.get_avg_sentiment()        

print(score_dict)

