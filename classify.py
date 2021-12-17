import sys
from transformers import pipeline


def classify(clf : pipeline, seq : str, labels : list, multi_class : bool=False) -> str:
    """
    Using classifier `clf`, returns the best label from `labels` for the sequence `seq`.
    If `multi_class` is set to True, returns a list of scores indexed to match `labels`.
    """
    return clf(seq, labels, multi_class=multi_class)

if __name__ == "__main__":
    clf = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')


