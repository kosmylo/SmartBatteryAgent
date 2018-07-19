# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import division
from __future__ import print_function
from __future__ import print_function

import random as r
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from environment import Environment
from options import getDefaultObject


# Note: you may need to update your version of future
# sudo pip install -U future
# Inspired by https://github.com/dennybritz/reinforcement-learning


class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1

    def partial_fit(self, X, Y):
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)


def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


class FeatureTransformer:
    number_of_samples = 1000
    use_rbf_sampling = False

    def __init__(self, env):
        scaler = StandardScaler()
        observation_examples = []
        for i in range(FeatureTransformer.number_of_samples):
            observation_examples.append(env.sample())
        scaler.fit(observation_examples)
        if FeatureTransformer.use_rbf_sampling:
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
            self.featurizer = featurizer
        else:
            self.dimensions = len(observation_examples[0])
            self.featurizer = None
        self.scaler = scaler

    def transform(self, observations):
        self.scaler.partial_fit(observations)
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled) if self.featurizer else scaled


# Holds one SGDRegressor for each action
class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.neg_value = -1000
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            self.models.append(SGDRegressor(feature_transformer.dimensions))

    def predict(self, s, legal_actions=None):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        predictions = np.array(
            [m.predict(X)[0] if (legal_actions is None or i in legal_actions) else self.neg_value for (i, m)
             in enumerate(self.models)])
        sum_predictions = sum([abs(prediction) for prediction in predictions])
        return [prediction / sum_predictions for prediction in predictions]

    def update(self, s, a, q_value):
        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(X, [q_value])

    def sample_action(self, s, eps):
        p = r.uniform(0, 1)
        if p < eps:
            action_index = self.env.action_space.sample(self.env.current_state)
        else:
            action_index = np.argmax(self.predict(s, self.env.action_space.get_legal_actions(
                self.env.current_state))) if env_options.use_legal_actions else np.argmax(self.predict(s))
        return action_index


def play_one(env, model, eps, gamma):
    state = env.reset()
    done = False
    total_reward = 0
    iters = 0
    number_of_hours_lasted = 0
    while not done and iters < 2000:
        # if we reach 2000, just quit, don't want this going forever
        # the 200 limit seems a bit early
        action_index = model.sample_action(state, eps)
        cached_current_state = state
        state, reward, done, info = env.step(action_index)
        if done:
            reward = -1000
            q_value = reward
        else:
            future_rewards = model.predict(state)
            q_value = reward + gamma * max(future_rewards)
        model.update(cached_current_state, action_index, q_value)

        if reward != -1000:  # if we changed the reward to -200
            total_reward += reward
        iters += 1
        number_of_hours_lasted += 1
    avg_reward = total_reward / number_of_hours_lasted
    return avg_reward, number_of_hours_lasted


def plot_savings(action_list, grid_list, solar_list, load_list, energy_list, price_list):
    plt.plot(action_list, label='action')
    plt.plot(grid_list, label='from grid')
    plt.plot(solar_list, label='solar power')
    plt.plot(load_list, label='required load')
    plt.plot(energy_list, label='stored energy')
    plt.plot(price_list, label='unit price')
    plt.xlabel('hours')
    plt.legend(loc='best')
    plt.show()


def get_savings(env, model, plot=False):
    observation = env.reset()
    grid_list = [[] for _ in range(env.day_chunk)]
    action_list = [[] for _ in range(env.day_chunk)]
    solar_list = [[] for _ in range(env.day_chunk)]
    netload_list = [[] for _ in range(env.day_chunk)]
    load_list = [[] for _ in range(env.day_chunk)]
    energy_list = [[] for _ in range(env.day_chunk)]
    number_of_hours_lasted = 0
    while env.day_number < (env.day_chunk - 1):
        action_index = model.sample_action(observation, 0)
        prev_observation = observation
        observation, reward, done, info = env.step(action_index)
        if done:
            break
        number_of_hours_lasted += 1
        p_grid = env.get_p_grid(prev_observation, action_index)
        grid_list[env.day_number - 1].append(p_grid)
        action_list[env.day_number - 1].append(env.action_space.actions[action_index])
        load_list[env.day_number - 1].append(prev_observation[1])
        solar_list[env.day_number - 1].append(prev_observation[0])
        energy_list[env.day_number - 1].append(prev_observation[2])
        netload_list[env.day_number - 1].append(prev_observation[1] - prev_observation[0])
    max_savings = -1000
    best_day = -1
    for day in range(env.day_chunk):
        agent_bill = sum([a * b for a, b in zip(env.price_scheme, grid_list[day])])
        base_bill = sum([max(0, a * b) for a, b in zip(env.price_scheme, netload_list[day])])
        savings = ((base_bill - agent_bill) * 100) / max(1, base_bill)
        if savings > max_savings:
            max_savings = savings
            best_day = day
    if plot and len(action_list[best_day]) >= 24:
        plot_savings(action_list[best_day], grid_list[best_day], solar_list[best_day],
                     load_list[best_day], energy_list[best_day], map(lambda x: x*50, env.price_scheme))
    return max_savings, number_of_hours_lasted


env_options = getDefaultObject()


def main():
    r.seed(int(time.time()))
    env = Environment()
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    N = 1000
    total_rewards = np.empty(N)
    number_of_hours_lasted_lst = np.empty(N)
    savings = []
    for n in range(N):
        eps = min(0.1, 1.0 / np.sqrt(n + 1))
        total_reward, number_of_hours_lasted = play_one(env, model, eps, env.env_options.gamma)
        total_rewards[n], number_of_hours_lasted_lst[n] = total_reward, number_of_hours_lasted
        if n > 0 and n % 50 == 0:
            current_saving, hours_lasted = get_savings(env, model, False)
            print('The average saving per hour for the best day for the model: ', current_saving, ', and it lasted for',
                  hours_lasted, 'hours')
            savings.append(current_saving)
        if n > 0 and n % 50 == 0:
            print('plotting savings after ', n, ' iterations')
            get_savings(env, model, False)
    if n % 100 == 0:
        print("episode:", n, "total reward:", total_reward, "eps:", eps, "avg reward (last 100):",
              total_rewards[max(0, n - 100):(n + 1)].mean())
    print("avg reward for last 100 episodes:", total_rewards[-100:].mean())
    print("avg number of hours lasted for last 100 episodes:", number_of_hours_lasted_lst[-100:].mean())
    print("total steps:", total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plt.plot(savings)
    plt.ylabel("savings per hour in %")
    plt.xlabel("number of iterations")
    plt.show()

    plt.plot(number_of_hours_lasted_lst)
    plt.ylabel("number of hours lasted per iteration")
    plt.xlabel("number of iterations")
    plt.show()


if __name__ == '__main__':
    main()
