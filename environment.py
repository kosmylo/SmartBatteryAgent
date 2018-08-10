import random as r

import numpy as np
import pandas as pd

from options import get_default_object

debug = not True


class Environment:
    '''
        state = [solar, load, energy_level, price, time_step]
    '''
    env_options = None

    def __init__(self, env_options):
        self.env_options = env_options
        self.eta = self.env_options.eta  # (battery efficiency)
        self.Gamma = self.env_options.gamma
        self.start = self.env_options.start  # pick season
        self.day_chunk = self.env_options.day_chunk
        self.training_time = self.env_options.total_years
        self.price_scheme = self.env_options.price_scheme
        self.df_solar = pd.read_csv(self.env_options.solar_data)
        self.df_solar = self.df_solar[self.start: self.start + self.day_chunk].reset_index()
        self.df_load = pd.read_csv(self.env_options.load_data)
        self.df_load = self.df_load[self.start: self.start + self.day_chunk].reset_index()
        self.current_state = None
        self.day_number = 0
        self.time_step = 0
        self.action_space = Environment.ActionSpace(self.env_options)

    class ActionSpace:

        def __init__(self, env_options):
            self.env_options = env_options
            self.actions = self.env_options.actions
            self.n = len(self.actions)

        def sample(self, current_state):
            legal_actions = self.get_legal_action_indices(current_state)
            return r.choice(legal_actions)

        def get_legal_action_indices(self, current_state):
            '''
                Calculate and return allowable action set
                Output: List of indices of allowable actions
            '''

            legal_action_indices = []
            for k in range(len(self.actions)):
                if self.is_action_legal(current_state, self.actions[k]):
                    legal_action_indices.append(k)

            return legal_action_indices

        def is_action_legal(self, current_state, action):
            current_solar = current_state[0]
            current_load = current_state[1]
            current_net_load = current_load - current_solar

            if action >= 0:
                p_charge, p_discharge = action, 0.0
            else:
                p_charge, p_discharge = 0.0, action

            p_grid = current_net_load + p_charge + p_discharge
            e_next = current_state[2] + self.env_options.eta * p_charge + p_discharge
            return (p_grid >= 0 or p_grid < 0) and self.env_options.E_min <= e_next <= self.env_options.E_max

    def reset(self, reset_day=False):
        initial_state = self.get_initial_state(0, self.env_options.E_init
        if self.current_state is None else self.current_state[2])
        self.current_state = initial_state
        self.time_step = 0
        if reset_day:
            self.day_number = 0
        return initial_state

    @staticmethod
    def sample():
        solar_sample = np.random.uniform(low=0, high=6)
        load_sample = np.random.uniform(low=2.0, high=4.8)
        energy_sample = np.random.uniform(low=1.7, high=6.0)
        price_sample = np.random.uniform(low=0.0, high=0.12)
        time_step = r.randint(0, 23)
        sample_state = [solar_sample, load_sample, energy_sample, price_sample, time_step]
        return sample_state

    def get_initial_state(self, day_number, e_init):
        '''
            Set's the initialState (0th hour) for day_number.
            day_number
        '''
        solar = float(self.df_solar[self.get_key(0)][day_number])
        load = float(self.df_load[self.get_key(0)][day_number])
        energy_level = e_init
        price = self.get_price(0)

        return [solar, load, energy_level, price, 0]

    def step(self, action_index):
        assert (action_index is not None)
        action = self.env_options.actions[action_index]
        next_state, reward, done, info = self.get_next_state(self.day_number, self.time_step, self.current_state,
                                                             action)
        self.time_step += 1
        if self.time_step > 23:
            self.day_number = self.day_number + 1
            self.time_step = 0
        if self.day_number >= self.day_chunk:
            self.day_number = 0
        self.current_state = next_state
        return next_state, reward, done, info

    def get_next_state(self, day_number, time_step, state_k, action_k):

        current_solar = state_k[0]
        current_load = state_k[1]
        current_energy = state_k[2]
        current_netload = current_load - current_solar

        if action_k >= 0:
            p_charge, p_discharge = action_k, 0.0
        else:
            p_charge, p_discharge = 0.0, action_k
        e_next = current_energy + self.eta * p_charge + p_discharge
        p_grid = current_netload + p_charge + p_discharge
        is_valid = True
        reward = self.get_non_myopic_reward_function(p_grid, time_step)

        if not is_valid:
            reward = -100
            next_state = None
        else:
            next_state = [self.get_solar(day_number, time_step + 1), self.get_load(day_number, time_step + 1), e_next,
                          self.get_price(time_step + 1), time_step + 1]
        return next_state, reward, (not is_valid), 'info is not supported'

    def get_reward(self, p_grid, time_step):
        return -p_grid * self.get_price(time_step)

    def get_non_myopic_reward_function(self, p_grid, time_step):
        current_price = self.get_price(time_step)
        reward = -current_price
        for price in [self.get_price(time) for time in range(time_step+1, 24)]:
            reward += (price - current_price)
        return reward * p_grid

    def get_price(self, time_step):
        return self.price_scheme[time_step % 24]

    @staticmethod
    def get_key(time_step):
        time_step = str(time_step)
        return time_step + ':00'

    def get_solar(self, day_number, time_step):
        if time_step > 23:
            day_number = day_number + 1
            time_step %= 24
        day_number = day_number % self.day_chunk
        time_step = self.get_key(time_step)
        return self.df_solar[time_step][day_number]

    def get_load(self, day_number, time_step):
        if time_step > 23:
            day_number = day_number + 1
            time_step %= 24
        day_number = day_number % self.day_chunk
        time_step = self.get_key(time_step)
        return self.df_load[time_step][day_number]

    def get_pgrid(self, state, action_index):
        assert (action_index is not None)
        action = self.env_options.actions[action_index]
        if action >= 0:
            p_charge, p_discharge = action, 0.0
        else:
            p_charge, p_discharge = 0.0, action
        p_grid = state[1] - state[0] + p_charge + p_discharge
        return p_grid


if __name__ == '__main__':
    '''
        code for testing the environment class
    '''
    environment = Environment()
    environment.reset()
