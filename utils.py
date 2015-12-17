#!/usr/bin/env python
"""Utility for processing raw HTML text into segments for further learning."""

import re

import nltk
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# Beautiful Soup throws a lot of benign warnings which we will ignore
import warnings
warnings.filterwarnings("ignore")


def text_to_wordlist(raw_text, remove_stopwords=False):
    """Function to convert a document to a sequence of words,
    optionally removing stop words.  Returns a list of words."""

    # Pre-processing. Removing HTML
    # TODO: Do further pre-processing (eg. remove links)
    text = BeautifulSoup(raw_text).get_text()

    # Remove non-letters
    text = re.sub("[^a-zA-Z]"," ", text)

    # Convert words to lower case and split them
    words = text.lower().split()

    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)


def text_to_w2v_input(text, tokenizer=None, remove_stopwords=False):
    """Function to split a text into parsed sentences.

    text: un-processed text, could be several sentences on one line.
    tokenizer: the NLTK sentence tokenizer to use. Punkt by default.
    remove_stopwords: get rid of English stop words. Not recommended for W2V.

    returns: list of tokenized sentences, each sentence is a list of words.
             This is the format required by W2V."""

    # NOTE: Punkt is a sentence tokenizer
    if not tokenizer:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Split text into sentences
    raw_sentences = tokenizer.tokenize(text.decode('utf8').strip())

    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if raw_sentence:
            tokenized_sentences.append(
                text_to_wordlist(raw_sentence, remove_stopwords))

    return tokenized_sentences
