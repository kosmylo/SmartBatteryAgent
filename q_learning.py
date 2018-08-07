# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import division
from __future__ import print_function
from __future__ import print_function

import random as r
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from environment import Environment
from options import get_default_object


# Note: you may need to update your version of future
# sudo pip install -U future
# Inspired by https://github.com/dennybritz/reinforcement-learning


def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


# Holds one SGDRegressor for each action
class Model:
    class FeatureTransformer:
        number_of_samples = 100
        use_rbf_sampling = False

        def __init__(self, env):
            scaler = StandardScaler()
            observation_examples = []
            for i in range(Model.FeatureTransformer.number_of_samples):
                observation_examples.append(env.sample())
            scaler.fit(observation_examples)
            if Model.FeatureTransformer.use_rbf_sampling:
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

    class SGDRegressor:
        def __init__(self, D):
            self.w = np.random.randn(D) / np.sqrt(D)
            self.lr = 0.1

        def partial_fit(self, X, Y):
            self.w += self.lr * (Y - X.dot(self.w)).dot(X)[0]

        def predict(self, X):
            return X.dot(self.w)

    class SklearnSGDRegressor:
        def __init__(self, alpha, state):
            self.regressor = SGDRegressor(alpha=alpha, max_iter=1000, tol=1e-3)
            self.regressor.partial_fit([state], [0])

        def partial_fit(self, X, Y):
            self.regressor.partial_fit(X, Y)

        def predict(self, X):
            return self.regressor.predict(X)

    def __init__(self, env):
        self.env = env
        self.models = []
        self.neg_value = -1000
        self.feature_transformer = Model.FeatureTransformer(env)
        for i in range(env.action_space.n):
            self.models.append(Model.SklearnSGDRegressor(0.1, env.reset()))

    def get_qvalue_for_actions(self, s):
        legal_actions = self.env.action_space.get_legal_action_indices(s)
        x = self.feature_transformer.transform(np.atleast_2d(s))
        predictions = np.array(
            [m.predict(x)[0] if (legal_actions is None or i in legal_actions) else self.neg_value for (i,
                                                                                                       m)
             in enumerate(self.models)])
        sum_predictions = sum([abs(prediction) for prediction in predictions])
        return [prediction / sum_predictions for prediction in predictions]

    def get_qvalue(self, state, action_index):
        return self.models[action_index].predict(self.feature_transformer.transform(np.atleast_2d(state)))

    def update(self, state, action_index, qvalue):
        x = self.feature_transformer.transform(np.atleast_2d(state))
        self.models[action_index].partial_fit(x, qvalue)

    def print_model_weights(self):
        for model in self.models:
            print(model.w)


class QLearning:

    def __init__(self, env, env_options, model, eps):
        self.env = env
        self.env_options = env_options
        self.model = model
        self.eps = eps

    @staticmethod
    def argmax(values):
        max_value = -1000
        for value in values:
            max_value = max(max_value, value)
        max_indices = []
        for i, value in enumerate(values):
            if value == max_value:
                max_indices.append(i)
        return r.choice(max_indices)

    def get_best_action_index(self, s):
        p = r.uniform(0, 1)
        if p < self.eps:
            action_index = self.env.action_space.sample(self.env.current_state)
        else:
            action_index = self.argmax(self.model.get_qvalue_for_actions(s))
        return action_index

    def get_new_qvalue(self, q_old, reward, q_max_next_state):
        q_new = (1 - self.env_options.learning_rate) * q_old + self.env_options.learning_rate * (reward +
                                                                                                 self.env_options.gamma *
                                                                                                 q_max_next_state)
        return q_new

    def update_gamma(self):
        #  starting gamma out with a lower value and increasing as the model starts to learn
        self.env_options.gamma += 0.0001
        self.env_options.gamma = min(0.8, env_options.gamma)

    def run_agent(self, max_allowed_hours=24*10):
        state = self.env.reset()
        done = False
        number_of_hours_lasted = 0
        action_indices = []
        rewards_per_episode = defaultdict(lambda: 0)
        episode_number = 0
        while not done and number_of_hours_lasted < max_allowed_hours:
            # if we reach 2000, just quit, don't want this going forever
            # the 200 limit seems a bit early
            action_index = self.get_best_action_index(state)
            action_indices.append(action_index)
            cached_prev_state = state
            state, reward, done, info = self.env.step(action_index)
            if done:
                q_value = [reward]
            else:
                future_rewards = self.model.get_qvalue_for_actions(state)
                q_value = self.get_new_qvalue(self.model.get_qvalue(cached_prev_state, action_index), reward,
                                              max(future_rewards))
                self.update_gamma()
            self.model.update(cached_prev_state, action_index, q_value)
            number_of_hours_lasted += 1
            episode_number = int(number_of_hours_lasted / 24)
            if not done:
                rewards_per_episode[episode_number] += reward
        avg_reward_per_episode = sum(rewards_per_episode.values()) * 1.0 / max(1, int(number_of_hours_lasted / 24))
        return avg_reward_per_episode, number_of_hours_lasted, rewards_per_episode


def plot_a_day(action_list, grid_list, solar_list, load_list, energy_list, price_list, net_load_list):
    plt.plot(action_list, label='action')
    plt.plot(grid_list, label='from grid')
    plt.plot(solar_list, label='solar power')
    plt.plot(load_list, label='required load')
    plt.plot(energy_list, label='stored energy')
    plt.plot(price_list, label='unit price')
    plt.plot(net_load_list, label='net load list')
    plt.xlabel('hours')
    plt.legend(loc='best')
    plt.show()


def plot_savings_and_rewards(savings_per_day, reward_per_episode):
    plt.plot(savings_per_day, label='savings per day')
    plt.plot(reward_per_episode, label='reward per episode')
    plt.legend(loc='best')
    plt.xlabel('# of days')
    plt.ylabel('savings % and total reward')
    plt.show()


def test_model_actual(env, model, plot=False):
    state = env.reset()
    grid_list = [[] for _ in range(env.day_chunk)]
    action_list = [[] for _ in range(env.day_chunk)]
    solar_list = [[] for _ in range(env.day_chunk)]
    net_load_list = [[] for _ in range(env.day_chunk)]
    load_list = [[] for _ in range(env.day_chunk)]
    energy_list = [[] for _ in range(env.day_chunk)]
    reward_per_day = [0] * env.day_chunk
    number_of_hours_lasted = 0
    q_learning = QLearning(env, env_options, model, 0)
    episode_number = 0
    while episode_number < env.day_chunk:
        action_index = q_learning.get_best_action_index(state)
        prev_state = state
        state, reward, done, info = env.step(action_index)
        if done:
            break
        pgrid = env.get_pgrid(prev_state, action_index)
        grid_list[env.day_number - 1].append(pgrid)
        print('action taken by the agent:', env.action_space.actions[action_index])
        action_list[env.day_number - 1].append(env.action_space.actions[action_index])
        load_list[env.day_number - 1].append(prev_state[1])
        solar_list[env.day_number - 1].append(prev_state[0])
        energy_list[env.day_number - 1].append(prev_state[2])
        net_load_list[env.day_number - 1].append(prev_state[1] - prev_state[0])
        reward_per_day[env.day_number - 1] += reward
        number_of_hours_lasted += 1
        episode_number = number_of_hours_lasted // 24
    max_savings = -1000
    best_day = -1
    savings_per_day = []
    negative_rewards_count = 0
    for day in range(env.day_chunk):
        agent_bill = sum([a * b for a, b in zip(env.price_scheme, grid_list[day])])
        base_bill = sum([max(0, a * b) for a, b in zip(env.price_scheme, net_load_list[day])])
        savings = ((base_bill - agent_bill) * 100) / max(1, base_bill)
        if savings < 0:
            # plot_a_day(action_list[day], grid_list[day], solar_list[day],
            #            load_list[day], energy_list[day], map(lambda x: x * 50, env.price_scheme), net_load_list[day])
            negative_rewards_count += 1
        if savings > max_savings:
            max_savings = savings
            best_day = day
        savings_per_day.append(savings)
    if plot and len(action_list[best_day]) >= 24:
        plot_a_day(action_list[best_day], grid_list[best_day], solar_list[best_day],
                   load_list[best_day], energy_list[best_day], map(lambda x: x * 50, env.price_scheme),
                   net_load_list[best_day])
        plot_savings_and_rewards(savings_per_day, reward_per_day)
    return max_savings, number_of_hours_lasted, sum(savings_per_day) * 1.0 / env.day_chunk, negative_rewards_count


env_options = get_default_object()


def test_model(savings, average_reward_per_episode, total_rewards_per_episode, eps, model, env, n):
    current_saving, hours_lasted, average_saving, negative_rewards_count = test_model_actual(env, model, True)
    savings.append(average_saving)
    print('The saving for the best day for the model: ', current_saving, 'and the average savings per day: ',
          average_saving, 'and it lasted for', hours_lasted * 1.0 / 24, ' days', 'got negative rewards for ',
          negative_rewards_count)
    print("iteration:", n, "average reward per episode:", average_reward_per_episode, "eps:", eps, "avg reward "
                                                                                                   "(last ",
          (n + 1 - max(0, n - 100)), '): episodes', sum(total_rewards_per_episode[max(0, n - 100):(n + 1)]) / (
                  n + 1 - max(0, n - 100)))
    return savings


def plot(values, y_label, x_label, title):
    plt.plot(values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_model_savings(total_rewards_per_episode, savings, number_of_hours_lasted_lst):
    plot(total_rewards_per_episode, "rewards", "days", "Rewards")

    plot(savings, "average savings per day in %", "number of iterations", "Average Savings")

    plot(number_of_hours_lasted_lst, "number of hours lasted per iteration", "number of iterations", "Hours Lasted")


def main():
    r.seed(int(time.time()))
    env = Environment(env_options)
    model = Model(env)
    N = 10000
    total_rewards = []
    number_of_hours_lasted_lst = np.empty(N)
    savings = []
    avg_rewards_per_episode_per_iteration = []
    qlearning = QLearning(env, env_options, model, 0)
    for n in range(N):
        eps = min(0.1, 1.0 / np.sqrt(n + 1))
        qlearning.eps = eps
        average_reward_per_episode, number_of_hours_lasted, rewards_per_episode = qlearning.run_agent()
        total_rewards += rewards_per_episode
        number_of_hours_lasted_lst[n] = number_of_hours_lasted
        avg_rewards_per_episode_per_iteration.append(average_reward_per_episode)
        if n > 0 and n % 50 == 0:
            test_model(savings, average_reward_per_episode, total_rewards, eps, model, env, n)
            # plot_model_savings(total_rewards, savings, number_of_hours_lasted)
            # model.print_model_weights()
            print('average reward per episode', average_reward_per_episode, 'number of hours lasted',
                  number_of_hours_lasted)
            plot(avg_rewards_per_episode_per_iteration, 'rewards', 'iteration', 'avg rewards per episode per iteration')

    total_rewards = np.array(total_rewards)
    print("avg reward for last ", min(100, len(total_rewards)), " episodes:",
          total_rewards[-min(100, len(total_rewards)):].mean())
    print("avg number of hours lasted for last ", min(100, len(total_rewards)), " episodes:",
          number_of_hours_lasted_lst[-min(100, len(total_rewards)):].mean())
    print("total steps:", total_rewards.sum())


if __name__ == '__main__':
    main()
