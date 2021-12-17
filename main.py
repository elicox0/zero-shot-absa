import sys
import re
import numpy as np
from typing import Generator

from transformers import pipeline
from . import lda, classify, sentiment


PREFIX = '/home/eli/Code/absalute-zero'
PATH_TO_DATA = f'{PREFIX}/reviews_yelp.json' 
NUM_TOPICS = 10 # 100 may be suitable for generic corpus of reviews; 20 or fewer for single business

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
    get_sentences: returns an iterator over all pertinent sentences
    get_sentiment: returns average sentiment for the topic
    """
    def __init__(self, lda_out : list[tuple]) -> None:
        self.words = [t[0] for t in lda_out] # t is tuple of (word, weight)

    def set_sentences(self, corpus : Generator[str,None,None]) -> None:
        """
        Sets `self.sentences` to a generator of all sentences in the corpus that 
        contain at least one  word in the list of words corresponding to the topic.
        """
        punc_re = re.compile(r'[\.?!]+')
        self.sentences = (seq for seq in (punc_re.split(doc) 
                    for doc in corpus) 
                    if any((re.search(word, seq) for word in topic)))

    def set_name(self, clf : pipeline, labels : list[str]) -> None:
        """
        Sets `self.name` to the best label from the list provided as determined 
        by a zero-shot classifier. `self.sentences` must be set already.
        """
        result = clf(sequence, labels)
        self.name = result['labels'][np.argmax(result['scores'])]
        labels.remove(self.name) # if we want to prevent duplicate labels

    def get_avg_sentiment(self) -> float:
        """
        Returns the average sentiment over the object's sentences.
        """
        clf = pipeline('sentiment-analysis')
        return np.mean((clf(s) for s in self.sentences))

clf = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
corpus_dict = lda.create_dictionary()
score_dict = {}
topics = lda.get_model_output(num_topics, corpus_dict)

for topic in topics:
    with open(PATH_TO_DATA) as handle:
        t = Topic(clf(), topic)
        sentences = sentiment.get_sentences(lda.json_to_text_it(handle), topic)
        score_dict[topic] = sentiment.avg_sentiment(sentences)        

