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
        features = Counter()  
        tokens = self.normalizer(sentence)
        for word in tokens:
            index = self.indexer.add_and_get_index(word, add=add_to_indexer)
            if index >= 0:
                features[index] += 1
        
        return features


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
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
        features = Counter()
        for i in range(len(sentence) - 1):
            bigram = f"Bigram={sentence[i]}__{sentence[i + 1]}"
            print(bigram)
            index = self.indexer.add_and_get_index(bigram, add=add_to_indexer)
            if index >= 0:
                features[index] += 1
        return features
                


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer, n: int = 5):
        self.indexer = indexer
        self.n = n
        
    def get_indexer(self):
        return self.indexer
    
    def normalizer(self, sentence: List[str]) -> List[str]:
        normalized = re.findall(r"[a-zA-Z0-9']+", ' '.join(sentence))
        normalized = [word.lower() for word in normalized]
        return normalized

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        tkns = self.normalizer(sentence)
        i = 0
        features = Counter()
        
        while i < len(tkns) - 1:
            for j in range(1, self.n + 1):
                n_gram = '__'.join(tkns[i:i + j])
                titled_ngram = f"{self.n}_gram={n_gram}"
                index = self.indexer.add_and_get_index(titled_ngram, add=add_to_indexer)
                if index >= 0:
                    features[index] += 1
            i += 1
        return features
    


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
        z=0.0
        for index, count in features.items():
                
                if index >= 0:
                    z += self.weights[index] * count

        z += self.bias

        y_prob = sigmoid(z)
        return 1 if y_prob > 0.5 else 0

    def predict_proba(self, ex_words: List[str]) -> float:
        """
        Return the probability of the positive class
        :param ex_words: words in the example to predict on
        :return: a float between 0 and 1
        """
        features = self.featurizer.extract_features(ex_words, add_to_indexer=False)
        z=0.0
        for index, count in features.items():
                
                if index >= 0:
                    z += self.weights[index] * count

        z += self.bias

        y_prob = sigmoid(z)
        return y_prob


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # Extract features
    
    for ex in train_exs:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=True)
        
        
    total_features = feat_extractor.get_indexer().__len__()
            

    # Train logistic regression
    
    weight_vector=np.zeros(total_features)
    b=0.0
    learning_rate = 0.1
    
    for epoch in range(20):
        total_loss=0# number of epochs
        random.shuffle(train_exs)
        
            
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)

            #print(sentence_count.items())
            # print(index_cache)]
            z=0.0
            for index, count in features.items():
                
                if index >= 0:
                    z += weight_vector[index] * count
                    
            z += b

            y_prob = sigmoid(z)
            loss = entropy_loss(ex.label, y_prob)
            total_loss += loss
    # Gradient descent step (you may want to implement mini-batch or full-batch gradient descent)

            error = y_prob - ex.label
            
            for index, count in features.items():
                weight_vector[index] -= learning_rate*error * count
            
            
            b -= learning_rate * error
        print(f"Epoch {epoch}, loss: {total_loss/len(train_exs)}") 

        learning_rate *= 0.9
        
    return LogisticRegressionClassifier(weight_vector, b, feat_extractor)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def entropy_loss(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


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
        self.network = network
        self.word_embeddings = word_embeddings
        
    def predict(self, ex_words: List[str]) -> int:
        total = np.zeros(self.word_embeddings.get_embedding_length())
        for word in ex_words:
            embedding = self.word_embeddings.get_embedding(word)
            total += embedding
        avg = total / len(ex_words)
        logits = self.network(torch.FloatTensor(avg))
        return 1 if torch.softmax(logits, dim=0).argmax(dim=0).item() > 0.5 else 0

    def predict_proba(self, ex_words: List[str]) -> float:
        """
        Return the probability of the positive class
        :param ex_words: words in the example to predict on
        :return: a float between 0 and 1
        """
        total = np.zeros(self.word_embeddings.get_embedding_length())
        for word in ex_words:
            embedding = self.word_embeddings.get_embedding(word)
            total += embedding
        avg = total / len(ex_words)
        logits = self.network(torch.FloatTensor(avg))
        return torch.softmax(logits, dim=1)
        


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    print(word_embeddings.get_embedding_length())
    network  = nn.Sequential(nn.Linear(word_embeddings.get_embedding_length(), word_embeddings.get_embedding_length()//15),
                              nn.ReLU(),
                              nn.Linear(word_embeddings.get_embedding_length()//15, 2)
                              )

    average_matrix = np.zeros((len(train_exs), word_embeddings.get_embedding_length()))
    labels = np.zeros(len(train_exs))
    for i, ex in enumerate(train_exs):

            total = np.zeros(word_embeddings.get_embedding_length())
            for word in ex.words:
                embedding = word_embeddings.get_embedding(word)
                total += embedding
            avg = total / len(ex.words)
            average_matrix[i] = avg
            labels[i] = ex.label

            X = torch.from_numpy(average_matrix).float()           
            y = torch.from_numpy(labels).long()
            
    for epoch in range(10000):
        lr = 0.001
        logits = network(X)
        loss = nn.CrossEntropyLoss()
        loss_calc = loss(logits, y)
        #print(f"Epoch {epoch}, loss: {loss_calc.item()}")
        optimizer = optim.Adam(network.parameters(), lr=lr)

        optimizer.zero_grad()
        loss_calc.backward()
        optimizer.step()

    return NeuralSentimentClassifier(network, word_embeddings)

    
