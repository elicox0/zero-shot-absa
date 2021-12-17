#!/usr/bin/env python
# main.py

import sys
import os
import re
import numpy as np
from tqdm import tqdm
from typing import Generator

import transformers
import ijson

import gensim
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


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
        self.sentences = []
        for i,doc in enumerate(corpus):
            for seq in PUNC_RE.split(doc):
                if any((re.search(word,seq) for word in self.words)):
                    self.sentences.append(seq)
#        self.sentences = (seq for seq in (PUNC_RE.split(doc) 
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
        score = lambda result : (-1 if result['label'] == 'NEGATIVE' else 1) * result['score']
        clf = transformers.pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        return np.mean(np.array([score(clf(s)[0]) for s in self.sentences]))

def clean(doc : str, lemmatizer : object, return_list : bool=False) -> str:
    """
    Returns doc normalized (lemmatized) and without stopwords or punctuation.

    Parameters
    ----------
    doc <str> : the path (name) of the json file containing reviews.

    Returns
    -------
    <str> : the normalized string without stopwords or punctuation
    """
    stop_free  = " ".join([w for w in doc.lower().split() if w not in STOP]) # Split by word
    punc_free  =  "".join([c for c in stop_free if c not in PUNC])           # Split by character
    normalized = [lemmatizer.lemmatize(word) for word in punc_free.split()]
    if return_list:
        return normalized
    return " ".join(normalized)

def json_to_text_it(handle : bytes, to_clean : bool=True, **kwargs) -> Generator[str,None,None]:
    """
    Iterator over cleaned text of json objects in file with contents in `handle`.

    Parameters
    ----------
    handle <bytes> : The file handle. Should be opened with mode='rb'.
    *Passes kwargs to clean().

    Returns
    -------
    <Generator>: yields text from reviews one at a time. If `clean` flag is set,
                 yields the cleaned text as defined in `clean` function.
    """
    jobject = ijson.parse(handle)
    if to_clean:
        return (clean(item[2], LEMMATIZER, **kwargs) for item in jobject if item[0] == 'item.item.text')
    else:
        return (item[2] for item in jobject if item[0] == 'item.item.text')

def text_it(handle: bytes, to_clean : bool=True, **kwargs) -> Generator[str,None,None]:
    """
    Iterator over text files, separated by newlines.

    Parameters
    ----------
    handle <bytes> : The file handle. Should be opened with mode='rb'.
    *Passes kwargs to clean().

    Returns
    -------
    <Generator>: yields text from reviews one at a time. If `clean` flag is set,
                 yields the cleaned text as defined in `clean` function.
    """
    if to_clean:
        return (clean(line, LEMMATIZER, **kwargs) for line in handle.readlines())
    return (line for line in handle.readlines())

def create_dictionary(corpus : str, dtype : str) -> gensim.corpora.Dictionary:
    """
    Creates a dictionary to pass to the gensim LDA model.
    """
    with open(corpus, mode=DTYPE_MAP[dtype][1]) as data_stream:
        if os.path.exists('lda_dictionary.pk'):
            print("Using cached dictionary `lda_dictionary.pk`")
            dictionary = gensim.corpora.Dictionary.load('lda_dictionary.pk')
        else:
            dct_generator = DTYPE_MAP[dtype][0](data_stream, to_clean=True, return_list=True)
            dictionary = gensim.corpora.Dictionary()
            print("Creating dictionary...")
            for review in tqdm(dct_generator):
                dictionary.add_documents([review])
            dictionary.save('lda_dictionary.pk')
            del dct_generator
            print(f"Finished creating dictionary with {len(dictionary.keys())} words\n")

        return dictionary

def get_ldamodel(num_topics : int, 
                 corpus     : Generator[str,None,None], 
                 dictionary : gensim.corpora.Dictionary,
                 dtype      : str,
                 save       : str=None) -> gensim.models.ldamulticore.LdaMulticore:
    """
    Returns the topics and associated words.
    """
    with open(corpus, mode=DTYPE_MAP[dtype][1]) as data_stream:
        doc_generator = DTYPE_MAP[dtype][0](data_stream)
    doc_term_matrix = [dictionary.doc2bow([doc]) for doc in doc_generator]
    ldamodel = gensim.models.ldamulticore.LdaMulticore # Can use gensim.models.ldamodel.LDAModel for safer, but not parallel, model
    print("Fitting LDA model...\n")
    model = ldamodel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, eta=0.9, alpha=0.01, passes=PASSES, workers=NUM_WORKERS)
    if save is not None:
        model.save(save)
    return model

    # BEST SO FAR: eta=0.9, alpha=0.01

def pipeline(doc : str, lda : gensim.models.ldamulticore.LdaMulticore) -> dict:
    """
    Returns a dictionary of (topic : avg_sentiment) pairs for doc.
    """
    pass

def pipeline_eval(topics : list[str], doc : str) -> list:
    """
    Returns a list of tuples (topic, polarity), one tuple for
    each sentence in `doc`. For evaluation against the xml data.
    """
    clf = transformers.pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    score = lambda result : (-1 if result['label'].upper() == 'NEGATIVE' else 1)
    dtype = doc.split('.')[-1]
    dct = create_dictionary(doc, dtype)
    model = get_ldamodel(len(topics), doc, dct, dtype)
    with open(doc) as fin:
        sequences = PUNC_RE.split(fin.read())
    return [(topics[model.get_document_topics(dct.doc2bow([seq]))[0][0]], score(clf(seq)[0])) for seq in sequences]


PREFIX = '/home/eli/Code/absalute-zero'
PATH_TO_DATA   = f'{PREFIX}/foursquare.txt' 
PATH_TO_TOPICS = f'{PREFIX}/topics.txt'
STOP = set(stopwords.words('english'))
PUNC = set(string.punctuation)
LEMMATIZER = WordNetLemmatizer()
PUNC_RE = re.compile(r'[\.?!]+')
PASSES = 20
NUM_WORKERS = 3 # set to (number of cpu cores available) -1

DTYPE_MAP = {
        'txt'  : (text_it, 'r'),
        'json' : (json_to_text_it, 'rb'),
        #        'csv'  : (None, 'r')
}

if __name__ == "__main__":
    candidate_labels = sys.argv[1:]
    dtype = PATH_TO_DATA.split('.')[-1]
    clf = transformers.pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    score_dict = {}
    corpus_dict = create_dictionary(PATH_TO_DATA, dtype=dtype)

    ldamodel = get_ldamodel(len(candidate_labels), PATH_TO_DATA, corpus_dict, dtype=dtype)

    topics = [t[1] for t in ldamodel.show_topics(num_topics=-1, formatted=False)]
    for t in topics:
        print(t, end='\n\n')
    exit

    for topic in topics:
        with open(PATH_TO_DATA, mode=DTYPE_MAP[dtype][1]) as handle:
            t = Topic(topic)
            t.set_sentences(DTYPE_MAP[dtype][0](handle))
            t.set_name(clf, candidate_labels)
            score_dict[t.name] = t.get_avg_sentiment()        

    print(score_dict)

