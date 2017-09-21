# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import division
# Note: you may need to update your version of future
# sudo pip install -U future
# Inspired by https://github.com/dennybritz/reinforcement-learning

# Works best w/ multiply RBF kernels at var=0.05, 0.1, 0.5, 1.0


from environment import Environment
from options import EnvironmentOptions
from options import getDefaultObject

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1

    def partial_fit(self, X, Y):
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)



def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

class FeatureTransformer:

    number_of_samples = 1000

    def __init__(self, env):
        scaler = StandardScaler()
        observation_examples = []
        for i in range(FeatureTransformer.number_of_samples) :
            observation_examples.append(env.sample())
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=1000)),
            ("rbf5", RBFSampler(gamma=0.6, n_components=1000)),
            ("rbf6", RBFSampler(gamma=0.3, n_components=1000))
        ])
        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))
        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)


# Holds one SGDRegressor for each action
class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
          model = SGDRegressor(feature_transformer.dimensions)
          self.models.append(model)

    def predict(self, s, legal_actions = None):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        return np.array([m.predict(X)[0] for m in self.models]) if legal_actions == None \
               else np.array([m.predict(X)[0] for (i, m) in enumerate(self.models) if i in legal_actions])

    def update(self, s, a, G):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(X, [G])

    def getLegalActions(self, currentState):
        '''
        Calculate and return allowable action set
        Output: List of indices of allowable actions
        '''
        energy_level = currentState[2]
        lower_bound = max(env_options.E_min - energy_level, -env_options.P_cap)
        upper_bound = min(env_options.E_max - energy_level, env_options.P_cap)

        max_bin = int(np.digitize(upper_bound, env_options.actions, right=True))
        min_bin = int(np.digitize(lower_bound, env_options.actions, right=True))

        legal_actions = []
        for k in range(min_bin, max_bin):
            legal_actions.append(k)
        return legal_actions

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s, self.getLegalActions(self.env.current_state))) if env_options.use_legal_actions else np.argmax(self.predict(s))


def play_one(env, model, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    number_of_hours_lasted = 0
    while not done and iters < 2000:
        # if we reach 2000, just quit, don't want this going forever
        # the 200 limit seems a bit early
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        if done:
            reward = -1000
            G = reward
        else :
            next = model.predict(observation)
            assert(len(next.shape) == 1)
            G = reward + gamma*np.max(next)
        model.update(prev_observation, action, G)

        if reward != -1000: # if we changed the reward to -200
          totalreward += reward
        iters += 1
        number_of_hours_lasted += 1
    return totalreward, number_of_hours_lasted

def plot_savings(action_list, grid_list, solar_list, netload_list, load_list, energy_list):
    plt.plot(action_list, label = 'action')
    plt.plot(grid_list, label = 'grid load')
    plt.plot(solar_list, label = 'solar power')
    #plt.plot(netload_list, label = 'net load')
    plt.plot(load_list, label = 'household load')
    #plt.plot(energy_list, label = 'battery energy')
    plt.xlabel('hours')
    plt.show()

def get_savings(env, model, plot = False):
    savings = 0.0
    done = False
    observation = env.reset()
    action_list = []
    grid_list = []
    solar_list = []
    netload_list = []
    load_list = []
    energy_list = []
    price_list = []
    while env.day_number < (env.day_chunk - 1):
        action = model.sample_action(observation, 0)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        if done :
            break
        P_grid = env.get_p_grid(prev_observation, action)
        action_list.append(env_options.actions[action])
        grid_list.append(P_grid)
        load_list.append(prev_observation[1])
        solar_list.append(prev_observation[0])
        energy_list.append(prev_observation[2])
        netload_list.append(prev_observation[1] - prev_observation[0])
        price_list.append(prev_observation[3])
    agent_bill = sum([a * b for a, b in zip(price_list, grid_list)])
    base_bill = sum([max(0, a * b) for a, b in zip(price_list, netload_list)])
    savings = (base_bill - agent_bill)/base_bill
    print savings
    if(plot) :
        return plot_savings(action_list, grid_list, solar_list, netload_list, load_list, energy_list)
	return savings

env_options = getDefaultObject()

def main():
    env = Environment()
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99
    N = 1000
    totalrewards = np.empty(N)
    number_of_hours_lasted_lst = np.empty(N)
    savings = []
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        print 'episode number : ', n
        totalreward, number_of_hours_lasted = play_one(env, model, eps, gamma)
        totalrewards[n], number_of_hours_lasted_lst[n] = totalreward, number_of_hours_lasted
        if n > 0 and n % 50 == 0 :
            savings.append(get_savings(env, model, False))
        if n > 0 and n % 10 == 0 :
            print 'plotting savings after ', n, ' iterations'
            get_savings(env, model, True)
    if n % 100 == 0:
        print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("avg number of hours lasted for last 100 episodes:", number_of_hours_lasted_lst[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plt.plot(savings)
    plt.ylabel("savings in %")
    plt.xlabel("number of hours")
    plt.show()

if __name__ == '__main__':
    main()
