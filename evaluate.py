import sys
import xml.etree.ElementTree as xml

import main


class Review:
    """
    Review as initialized from XML tree.
    """
    def __init__(self, tree):
        self.xml_review = tree
        self.id = tree.attrib['rid']
        self.sentence = [Sentence(ss[0]) for ss in tree][0]

class Sentence:
    """
    Sentence as initialized from XML tree.
    """
    def __init__(self, tree):
        self.xml_sentence = tree
        self.text = tree[0].text
        try:
            self.opinion = Opinion(tree[1][0]) # Only keep the first opinion; difficult to evaluate on multiple topics.
        except IndexError:
            self.opinion = Opinion(None)

class Opinion:
    """
    Opinion as initialized from XML tree.
    """
    def __init__(self, tree):
        if tree == None:
            self.label = 'NA'
            self.polarity = 'NA'
        else:
            self.xml_opinion = tree
            self.label = tree.attrib['category']
            self.polarity = (1 if tree.attrib['polarity'].lower() == 'positive' else -1)

def evaluate_model(model_output : list[tuple], test_map : list[tuple]) -> float:
    """
    Returns the score for model_output on the foursquare dataset, as defined by

            (# sentences labeled correctly) * (# sentences with correct polarity)
    score = ---------------------------------------------------------------------
                                (# total sentences)^2

    This is used as a metric because normal metrics for classification or regression won't 
    work on zero-shot-absa, since
    1) LDA is unsupervised, and
    2) There are 2 parts to "correctly" classifying a sentence, namely getting the aspect
       label right and giving the right polarity score (positive or negative).
    """
    n2_total_sentences = len(model_output)**2
    n_labels = 0
    n_polarity = 0
    checks = 0  # Number of times tuples are compared, making sure they aren't all NA
    for comparison in zip(model_output, test_map):
        yhat, y = comparison
        if any(y) == 'NA':
            continue
        else:
            checks += 1
        if yhat[0] == y[0]:
            n_labels += 1
        if yhat[1] == y[1]:
            n_polarity += 1
    return (n_labels * n_polarity)/checks, n_labels, n_polarity, n2_total_sentences, checks


if __name__ == "__main__":
    with open('../foursquare_gold.xml') as fin:
        parser = xml.parse(fin)

    root = parser.getroot()
    reviews = [Review(child) for child in root]

    with open('../classes.txt') as fin:
        topics = [line.strip() for line in fin.readlines()]

    model_output = main.pipeline_eval(topics, '../foursquare.txt')
    test_map = [(r.sentence.opinion.label.replace("#"," "), r.sentence.opinion.polarity) for r in reviews]

    print(evaluate_model(model_output, test_map))
