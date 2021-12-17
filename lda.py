import sys
import os
import re
import csv
import ijson

from tqdm import tqdm
from typing import Generator

import gensim
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


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

def create_dictionary(corpus : str, dtype : str='text') -> gensim.corpora.Dictionary:
    """
    Creates a dictionary to pass to the gensim LDA model.
    """
    with open(corpus, mode=ITER_MAP[dtype][1]) as data_stream:
        dct_generator = ITER_MAP[dtype][0](data_stream, to_clean=True, return_list=True)
        dictionary = gensim.corpora.Dictionary()
        for review in tqdm(dct_generator):
            dictionary.add_documents([review])
        dictionary.save('lda_dictionary.pk')
        del dct_generator
        print(f"Finished creating dictionary with {len(dictionary.keys())} words\n")
        return dictionary

def get_model_output(num_topics : int, 
                     corpus     : Generator[str,None,None], 
                     dictionary : gensim.corpora.Dictionary,
                     dtype      : str='text') -> list:
    """
    Returns the topics and associated words.
    """
    with open(corpus, mode=ITER_MAP[dtype][1]) as data_stream:

        doc_generator = ITER_MAP[dtype][0](data_stream)
        doc_term_matrix = iter([dictionary.doc2bow([doc]) for doc in doc_generator])

        ldamodel = gensim.models.ldamulticore # Can use gensim.models.ldamodel.LDAModel for safer, but not parallel, model
        print("Fitting LDA model...\n")
        model = ldamodel.LdaMulticore(tqdm(doc_term_matrix), num_topics=num_topics, id2word=dictionary, passes=PASSES, workers=NUM_WORKERS)
    return model.print_topics(num_topics=-1)


PATH_TO_CORPUS = '/home/eli/Code/absalute-zero/wik.txt'
STOP = set(stopwords.words('english'))
PUNC = set(string.punctuation)
LEMMATIZER = WordNetLemmatizer()
PASSES = 20
NUM_WORKERS=3 # set to (number of cpu cores available) -1
ITER_MAP = {
        'text' : (text_it, 'r'),
        'json' : (json_to_text_it, 'rb'),
        'csv'  : None,
}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        num_topics = 10
    else:
        num_topics = sys.argv[1]
    d = create_dictionary(PATH_TO_CORPUS)
    print(get_model_output(num_topics, PATH_TO_CORPUS, d))

