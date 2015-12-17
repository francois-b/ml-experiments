"""Vectorizers based on Word2Vec."""

import os
import logging
import pickle
from hashlib import md5

import numpy as np
from sklearn.base import TransformerMixin
from gensim.models import Word2Vec

from utils import text_to_w2v_input, text_to_wordlist

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_w2v_model(sentences, num_features=300, min_word_count=40,
                  num_workers=4, context=10, downsampling=1e-3):
    """Return a trained Word2Vec model.

    sentences: a list of words
    min_word_count: winimum word count
    num_workers:    number of threads to run in parallel
    context:        context window size
    downsampling:   downsample setting for frequent words
    """

    logger.debug("Trying to get the W2V model with {} features, and begins "
        "with '{}'.".format(num_features, " ".join(sentences[0][:5])))
    model_hash = md5(str(sentences) + str(num_features)).hexdigest()
    model_filename = model_hash + ".w2v.model"

    if os.path.isfile(model_filename):
        logger.debug("Found the model in file '{}'.".format(model_filename))
        model = Word2Vec.load(model_filename)
    else:
        logger.debug("Didn't find the model.")
        logger.debug("Training Word2Vec model with {} sentences and "
            "{} features...".format(len(sentences), num_features))
        model = Word2Vec(sentences, workers=num_workers,
                         size=num_features, min_count=min_word_count,
                         window=context, sample=downsampling, seed=1)
        logger.debug("...done.")
        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)
        logger.debug("Saving model in {}.".format(model_filename))
        model.save(model_filename)

    return model


class W2vAvgVectorizer(TransformerMixin):
    """Vectorizer which averages the word vectors in a paragraph.

    A paragraph can be a document. TODO: improve naming."""

    def __init__(self, num_features):
        self.num_features = num_features
        self.w2v_model = None

    def _make_feature_vec(self, word_list):
        """Average all of the word vectors in a given article.

        word_list: list of words (strings)
        """

        # Pre-initialize an empty numpy array (for speed)
        feature_vec = np.zeros((self.num_features,), dtype="float32")

        # index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed.
        index2word_set = set(self.w2v_model.index2word)

        # Loop over each word in the word_list and, if it is in the model's
        # vocabulary, add its feature vector to the total
        nwords = 0
        for word in word_list:
            # NOTE: Careful there, if all words are in caps in the article,
            # this function will return nan values and blow up the forest.
            word = word.lower()
            if word in index2word_set:
                nwords += 1
                feature_vec = np.add(feature_vec, self.w2v_model[word])

        # Divide the result by the number of words to get the average
        feature_vec = np.divide(feature_vec, nwords)
        return feature_vec


    def transform(self, strings):
        """Given a list of strings (each one just un-processed text), calculate
        the average feature vector for each one and return a 2D numpy array.

        strings: list of strings
        return: a list of feature vectors """

        logger.debug("Converting {} strings into lists of "
            "sentences.".format(len(strings)))

        tokenized_strings = []
        for text in strings:
            tokenized_strings.append(text_to_wordlist(text, remove_stopwords=True))

        # Pre-allocate a 2D numpy array, for speed
        feature_vecs = np.zeros((len(tokenized_strings), self.num_features),
                                     dtype="float32")

        # Loop through the strings
        for counter, word_list in enumerate(tokenized_strings):

            # Call the function (defined above) that makes average feature vectors
            feature_vecs[counter] = self._make_feature_vec(word_list)

            # For DEBUG only
            if np.isnan(feature_vecs[counter][0]):
                import ipdb;ipdb.set_trace()


        return feature_vecs


    def fit(self, examples):
        """Convert examples into lists of sentences and train W2V model.

        examples: list of strings

        returns: None. Sets the W2V model on this object.
        """

        sentences = []
        for example in examples:
            sentences += text_to_w2v_input(example)

        self.w2v_model = get_w2v_model(sentences)

