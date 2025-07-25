# qlearningAgents.py
# ------------------
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


from calendar import leapdays
from csv import QUOTE_NONNUMERIC
from re import A
from Helpers.game import *
from Helpers.learningAgents import ReinforcementAgent
from Helpers.featureExtractors import *

import random, Helpers.util, math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - getQValue
        - computeValueFromQValues
        - computeActionFromQValues
        - getAction
        - update

      Instance variables you have access to:
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
        - self.qVals (the Q-Table)

      Functions you will use:
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.eval = False
        self.qVals = {}
        self.action_qVals = {}
        self.actions = ["North", "South", "East", "West", "Stop", "Exit"]
        

    def getQValue(self, state, action):
        action = action.capitalize()
        """
          Returns the Q-value of the (state,action) pair.
          Should return 0 if we have never seen the state,
          and initialize that state in the Q-Table.
        """
        "*** YOUR CODE HERE ***"

        # #print(self.qVals[state])

        # return self.qVals[state][action]

        if state not in self.qVals:
          self.qVals[state] = { 'North': 0 , 'South': 0, 'East': 0, 'West': 0, 'Stop': 0, 'Exit': 0 }
        
        return self.qVals[state][action]

      
    def computeValueFromQValues(self, state):
        """
          Returns the Q-value of the best action to take in this state.  
          Note that if there are no legal actions, which is the case at 
          the terminal state, you should return a value of 0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
          return 0.0
        
        max_q = float("-inf")
        for action in legalActions:
          q = self.getQValue(state, action)
          if q > max_q:
            max_q = q
        
        #return self.getQValue(state, self.computeActionFromQValues(state))
        return max_q



    def computeActionFromQValues(self, state):
        """
          Returns the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
          return None

        best_action = [None]
        max_q = float("-inf")

        for action in legalActions:
          q = self.getQValue(state, action)
          if q > max_q:
            max_q = q
            best_action = [action]
          elif q == max_q:
            best_action.append(action)
        
        return random.choice(best_action)
        


    def getAction(self, state):
        """
          Select the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          Use random.random() to generate a random number between 0-1
          Use random.choice(list) to pick a random item from a list
        """
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not legalActions:
          return None

        num = random.random() 
        if num <= self.epsilon:
          action = random.choice(legalActions)
          return action
        else:
          action = self.computeActionFromQValues(state)
          return action


    def update(self, state, action, nextState, reward):
        action = action.capitalize()
        """
          You should do your Q-Value updates here.

          NOTE: You should never call this function,
          it will be called for you.
        """
        "*** YOUR CODE HERE ***"
        curr_qVal = self.getQValue(state, action)
        next_qVal = self.computeValueFromQValues(nextState)
        new_qVal = self.alpha * (reward + (self.discount * next_qVal) - curr_qVal)
        self.qVals[state][action] += new_qVal
    

    #************Do Not Touch Anyting Below***************

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


