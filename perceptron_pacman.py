# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
# ----------


# Perceptron implementation for imitation learning
from distutils import core
from pyexpat import features
import Helpers.util
from Exercises.perceptron_multiclass import PerceptronClassifier
from random import uniform
from pacman import GameState


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.epochs = maxIterations
        self.weights = [] 
        

    def convert_data(self, data):
        # fix datatype issues
        # if it comes in inside of a list, pull it out of the list
        if isinstance(data, list):
            data = data[0]

        #data comes as a tuple
        all_moves_features = data[0] #grab the features (dict of action->features)
        legal_moves = data[1]        #grab the list of legal moves from this state
        return_features = {}
        #loop each action
        for key in all_moves_features:
            #convert feature values from dict to list
            all_features = ['foodCount', 'STOP', 'nearest_ghost', 'ghost-0', 'capsule-0', 'food-0', 'food-1', \
                            'food-2', 'food-3', 'food-4', 'capsule count', 'win', 'lose', 'score']
            dict_features = all_moves_features[key] 
            list_features = []
            # grab all feature values & put them in a list
            for feat in all_features:
                if feat not in dict_features:
                    list_features.append(0)
                else:
                    list_features.append(dict_features[feat])
            return_features[key] = list_features

        return (return_features, legal_moves) 


    def classify(self, data):
        #leave this call to convert_data here!
        features, legal_moves = self.convert_data(data)

        ##your code goes here##    
        best_score = -1000000000
        best_move = None

        for move in legal_moves:
            score = sum(w * x for w, x in zip(self.weights, features[move]))
            if score > best_score:
                best_score = score
                best_move = move
            
        return [best_move] 


    def train(self, train_data, labels):
        self.weights = []
        for i in range(len(self.legalLabels)):
            self.weights.append(uniform(-1, 1))

        for epoch in range(self.epochs):
            for i in range(len(train_data)):
                features, legal_moves = self.convert_data(train_data[i])
                actual = labels[i]

                predicted = self.classify(train_data[i])[0]

                if predicted != actual:
                    actual = features[actual]
                    predicted = features[predicted]

                    for j in range(len(self.weights)):
                        self.weights[j] += actual[j]
                        self.weights[j] -= predicted[j]
