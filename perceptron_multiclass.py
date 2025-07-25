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


# Perceptron implementation
import Helpers.util
from random import uniform


class PerceptronClassifier:

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.epochs = max_iterations
        self.weights = None


    def classify(self, data):
        prediction = []
        
        sums = []
        for i in range(len(self.legalLabels)):
            total_score = 0
            for j in range(len(data)):
                total_score += self.weights[i][j] * data[j]
            sums.append(total_score)

        prediction.append(sums.index(max(sums)))
        
        return prediction
      
 

    def train(self, train_data, labels):
        self.weights = []
        
        for _ in range(len(self.legalLabels)):
            class_weights = []
            for j in range(len(train_data[0])):
                class_weights.append(uniform(-1,1))
            self.weights.append(class_weights)
            

        for epoch in range(self.epochs):
            correct = 0
            total = 0

            for row, actual in zip(train_data, labels):
                predicted = self.classify(row)[0]
        
                if predicted != actual:
                    for j in range(len(row)):
                        self.weights[predicted][j] -= row[j]
                        self.weights[actual][j] += row[j]
                else:
                    correct += 1
                total +=1

            accuracy = (correct/ total) * 100
            print(f"Epoch {epoch} .... {accuracy:.1f}% correct.")
        return self.weights

        

    

