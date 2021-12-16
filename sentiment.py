import re
import numpy as np
from typing import Generator
from transformers import pipeline

def get_sentences(corpus : Generator[str,None,None], topic : list[str]) -> list[str]:
    """
    Returns a list of all sentences in the corpus that contain at least one
    word in the list of words (`topic`) corresponding to the topic.
    """
    punc_re = re.compile(r'[\.?!]+')
    return [seq for seq in (punc_re.split(doc) 
                for doc in corpus) 
                if any((re.search(word, seq) for word in topic))]

def avg_sentiment(sentences : list[str]) -> float:
    """
    Returns the average sentiment over given sentences.
    """
    clf = pipeline('sentiment-analysis')
    return np.mean([clf(s) for s in sentences])

