
from math import gamma
from Exercises.qlearningAgents import QLearningAgent
import Helpers.util
from Helpers.featureExtractors import *



class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
        ApproximateQLearningAgent
        Only need to override getQValue and update
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = Helpers.util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = Helpers.util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Q(state, action) = w * f(s,a) = sum_i w_i * f_i(s,a)
        """
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0.0
        for feature, value in features.items():
            q_value += self.weights[feature] * value
        return q_value


    def update(self, state, action, nextState, reward):
        """
        Update weights using:
        difference = (reward + gamma * max Q(s',a')) - Q(s,a)
        weight[feature] += alpha * difference * feature_value
        """
        features = self.featExtractor.getFeatures(state, action)
        next_actions = self.getLegalActions(nextState)

        max_q_next = 0.0
        if next_actions:
            max_q_next = max([self.getQValue(nextState, a) for a in next_actions])

        current_q = self.getQValue(state, action)
        difference = (reward + self.discount * max_q_next) - current_q

        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value

    def final(self, state):
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            print("Final Weights after training:")
            for feature, weight in self.weights.items():
                print(f"{feature}: {weight:.4f}")
