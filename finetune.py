import sys
import os

import gensim
import torch
import transformers
from transformers import pipeline

clf = pipeline("zero-shot-classification")
clf()
