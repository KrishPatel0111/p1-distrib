# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter
import re


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def normalizer(self, sentence: List[str]) -> List[str]:
        normalized = re.findall(r"[a-zA-Z0-9']+", ' '.join(sentence))
        normalized = [word.lower() for word in normalized]
        return normalized

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter(self.normalizer(sentence))  # simple bag of words
        if add_to_indexer:
            for word in features:
                self.indexer.add_and_get_index(word)
        return features


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weight_vector: np.ndarray, b: float = 0.0, featurizer: FeatureExtractor = FeatureExtractor()):
        self.weights = weight_vector  # numpy array
        self.bias = b
        self.featurizer = featurizer

    def predict(self, ex_words: List[str]) -> int:
        features = self.featurizer.extract_features(ex_words, add_to_indexer=False)
        x = np.zeros((1, self.featurizer.get_indexer().__len__()))
        for feat, count in features.items():
            feat_index = self.featurizer.get_indexer().index_of(feat)
            if feat_index >= 0:
                x[0, feat_index] = count

        y_pred = sigmoid(np.dot(x, self.weights) + self.bias)
        return 1 if y_pred > 0.5 else 0

    def predict_proba(self, ex_words: List[str]) -> float:
        """
        Return the probability of the positive class
        :param ex_words: words in the example to predict on
        :return: a float between 0 and 1
        """
        features = self.featurizer.extract_features(ex_words, add_to_indexer=False)
        x = np.zeros((1, self.featurizer.get_indexer().__len__()))
        for feat, count in features.items():
            feat_index = self.featurizer.get_indexer().index_of(feat)
            if feat_index >= 0:
                x[0, feat_index] = count
        return sigmoid(np.dot(x, self.weights) + self.bias)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # Extract features
    X_counter = []
    y = []
    for ex in train_exs:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=True)
        X_counter.append(features)
        y.append(ex.label)
        
    total_features = feat_extractor.get_indexer().__len__()
    X = np.zeros((len(train_exs), total_features))
    for i, features in enumerate(X_counter):
        for feat, count in features.items():
            feat_index = feat_extractor.get_indexer().index_of(feat)
            X[i, feat_index] = count

    # Train logistic regression
    weight_vector=np.zeros(total_features)
    b=0.0
    learning_rate = 0.1
    for epoch in range(10000):  # number of epochs
        A = np.dot(X, weight_vector) + b
        y_pred = sigmoid(A)
        loss = entropy_loss(y, y_pred)
    # Gradient descent step (you may want to implement mini-batch or full-batch gradient descent)
        y = np.asarray(y, dtype=float)
        grad_weight = np.dot(X.T, (y_pred - y)) / len(y)
        grad_b = np.sum(y_pred - y) / len(y)
        weight_vector -= learning_rate * grad_weight
        b -= learning_rate * grad_b
        if epoch % 1000 == 0:
            learning_rate *= 0.9  # decay learning rate
        
        

    return LogisticRegressionClassifier(weight_vector, b, feat_extractor)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def entropy_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        raise NotImplementedError


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    raise NotImplementedError
