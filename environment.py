from copy import deepcopy
import pandas as pd
from options import EnvironmentOptions
import random as r
import numpy as np
from options import getDefaultObject

debug = not True


class Environment:
    '''
        state = [solar, load, energy_level, price, time_step]
    '''
    env_options = getDefaultObject()

    def __init__(self):
        self.eta = Environment.env_options.eta  # (battery efficiency)
        self.Gamma = Environment.env_options.gamma
        self.start = Environment.env_options.start  # pick season
        self.day_chunk = Environment.env_options.day_chunk
        self.training_time = Environment.env_options.total_years
        self.price_scheme = Environment.env_options.price_scheme
        self.df_solar = pd.read_csv(Environment.env_options.solar_data)
        self.df_solar = self.df_solar[self.start: self.start + self.day_chunk]
        self.df_load = pd.read_csv(Environment.env_options.load_data)
        self.df_load = self.df_load[self.start: self.start + self.day_chunk]
        self.current_state = None
        self.day_number = 0
        self.time_step = 0
        self.action_space = Environment.ActionSpace(Environment.env_options.actions)

    class ActionSpace:

        def __init__(self, actions):
            self.n = len(actions)
            self.actions = actions

        def sample(self):
            return self.actions.index(r.choice(self.actions))

    def reset(self):
        initialState = self.getState(0, Environment.env_options.E_init)
        self.current_state = initialState
        return initialState

    def sample(self):
        solar_sample = np.random.uniform(low=0, high=2.5)
        load_sample = np.random.uniform(low=2.0, high=4.8)
        energy_sample = np.random.uniform(low=1.7, high=6.0)
        price_sample = np.random.uniform(low=0.0, high=0.12)
        time_step = r.randint(0, 23)
        sample_state = [solar_sample, load_sample, energy_sample, price_sample, time_step]
        return sample_state

    def getState(self, day_number, E_init):
        '''
            Set's the initialState (0th hour) for day_number.
            day_number
        '''
        solar = float(self.df_solar[self.get_key(0)][day_number])
        load = float(self.df_load[self.get_key(0)][day_number])
        energy_level = E_init
        price = self.get_price(0)

        initialState = [solar, load, energy_level, price, 0]
        return initialState

    def step(self, action):
        assert (action != None)
        action = Environment.env_options.actions[action]
        next_state = self.get_next_state(self.day_number, self.time_step, self.current_state, action)
        self.time_step += 1
        if self.time_step > 23:
            self.day_number = self.day_number + 1
            self.time_step %= 24
        if self.day_number >= self.day_chunk:
            self.day_number %= self.day_chunk
        self.current_state = next_state[0]
        return next_state

    def get_next_state(self, day_number, time_step, state_k, action_k):

        current_solar = state_k[0]
        current_load = state_k[1]
        current_energy = state_k[2]
        current_netload = current_load - current_solar

        if action_k >= 0:
            P_charge, P_discharge = action_k, 0.0
        else:
            P_charge, P_discharge = 0.0, action_k
        E_next = current_energy + self.eta * P_charge + P_discharge
        P_grid = current_netload + P_charge + P_discharge
        is_valid = (P_grid > 0)
        reward = -P_grid * self.get_price(time_step)

        if not is_valid:
            reward = -1000
            next_state = None
        else:
            next_state = [self.get_load(day_number, time_step + 1), self.get_solar(day_number, time_step + 1), E_next,
                          self.get_price(time_step + 1), time_step + 1]
        return next_state, reward, (not is_valid), 'info is not supported'

    def get_price(self, time_step):
        return self.price_scheme[time_step % 24]

    def get_key(self, time_step):
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

    def get_p_grid(self, state, action):
        assert (action != None)
        action = Environment.env_options.actions[action]
        if action >= 0:
            P_charge, P_discharge = action, 0.0
        else:
            P_charge, P_discharge = 0.0, action
        P_grid = state[1] - state[0] + P_charge + P_discharge
        return P_grid


if __name__ == '__main__':
    '''
		code for testing the environment class
	'''
    environment = Environment()
    environment.reset()
    print environment.step(Environment.env_options.actions[0])
