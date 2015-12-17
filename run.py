#!/usr/bin/env python

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score

from vectorizers.word2vec import W2vAvgVectorizer


if __name__ == '__main__':
    cli = """Run a text classifier based on word vector features.

Usage: run.py --labeled=<FILE1> --unlabeled=<FILE2> [--show-stats]

--labeled=<FILE1>    TSV file with the following headers: id, sentiment, review
--unlabeled=<FILE2>  TSV with headers: id, review
--show-stats         Display stats

-h --help            Display this message

"""

    from docopt import docopt

    arguments = docopt(cli)

    labeled_file = arguments["--labeled"]
    unlabeled_file = arguments["--unlabeled"]

    # Set up logging. This module is not meant to be imported, so set as root.
    logger = logging.getLogger("")
    handler = logging.FileHandler("run.log")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Started run.py with the following arguments: {}".format(
        arguments))

    train_df = pd.read_csv(labeled_file, header=0, delimiter="\t", quoting=3)
    train_articles = train_df["review"]
    train_sentiment = train_df["sentiment"]
    unlabeled_df = pd.read_csv(unlabeled_file, header=0, delimiter="\t",
                               quoting=3)
    unlabeled_articles = unlabeled_df["review"]

    logger.debug("The labeled set has {} samples".format(len(train_df)))
    logger.debug("The unlabeled set has {} samples".format(len(unlabeled_df)))

    test_size = 0.2
    logger.debug("Splitting the labeled dataset into {}% train and {}% test"
                 "".format((1 - test_size) * 100, test_size * 100))
    X_train, X_test, y_train, y_test = train_test_split(train_articles,
                                                        train_sentiment,
                                                        test_size=test_size)

    logger.debug("The training set has {} samples".format(len(X_train)))
    logger.debug("The test set has {} samples".format(len(X_test)))

    w2vv = W2vAvgVectorizer(300)

    # Without target articles in the w2v model, the data we pass to the random
    # forest classifier might have NANs!
    all_articles = pd.concat([train_articles, unlabeled_articles])

    logger.debug("Fitting the W2vAvgVectorizer model...")

    w2vv.fit(all_articles)

    logger.debug("Transforming with the W2vAvgVectorizer model...")

    X = w2vv.transform(X_train)

    logger.debug("...done.")
    logger.debug("Training a Random Forest classifier...")

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(X, y_train)

    logger.debug("...Classifier trained with {} examples".format(len(X)))

    # TODO: save the forest model
    logger.debug("Testing the model on {} test samples".format(len(X_test)))

    y_features = w2vv.transform(X_test)
    y_pred = forest.predict(y_features)

    recall = recall_score(y_test, y_pred)#, average='macro')
    precision = precision_score(y_test, y_pred)

    logger.info("Recall: {}".format(recall))
    logger.info("Precision: {}".format(precision))

    if arguments["--show-stats"]:
        print "Recall: {}".format(recall)
        print "Precision: {}".format(precision)
