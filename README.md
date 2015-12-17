# Sentiment analysis with Word2Vec

## Coding principles

It should be easy to identify the different steps in this example, clearly see the inputs and outputs of each steps, and be able to reuse them.

When a data structure is becoming too complex to track, it should be wrapped inside a class. For example, we are often speaking of "list of lists of single words", which might have been processed to remote stop words or not.
Variables that hold this structure might be named "sentences", "word_lists" and so on, but it is unclear what we're talking about.

Command line interfaces should be separated from reusable code.

A choice was made to put a lot of log statements in the code so it's easier to understand what the code is doing by running it, debug it and compare runs with different parameters.

## Installing & running the code

There's a requirements file with all necessary packages:

```bash
pip install -r requirements.txt
```

The CLI help of run.py explains all options to run the code.

## Inspiration

The idea is based on this example from Kaggle:
https://www.kaggle.com/c/word2vec-nlp-tutorial
