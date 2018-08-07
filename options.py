import numpy as np


class EnvironmentOptions:

    def __init__(self, eta, gamma, start, day_chunk, total_years, price_scheme, look_ahead, E_cap, P_cap, E_init,
                 epsilon, actions, solar_data, load_data, learning_rate):
        '''
            options class from where environment get's all it's parameters.
            can be custom created or a default veriosn can also be loaded.
            added to provide flexibility as well as make the process of creating
            the environment uniform, like other openai-gym environments
		'''
        self.eta = eta  # battery efficiency
        self.gamma = gamma  # discount factor, importance given to future Q value predicted by the model
        self.start = start  # starting index in training files
        self.day_chunk = day_chunk  # no of days(elements) to consider in the data files
        self.total_years = total_years  # how many times do we want to repeat over the data
        self.price_scheme = price_scheme  # what price scheme per hour we are using
        self.E_cap = E_cap  # battery energy capacity (confirm)
        self.P_cap = P_cap  # battery power capacity (confirm)
        self.E_init = E_init  # initial energy of the battery
        self.epsilon = epsilon  # what is this ?
        self.actions = actions  # what are the actions to be taken by the agent ?
        self.look_ahead = look_ahead  # how many steps in time when we want to look ahead
        self.solar_data = solar_data  # solar data path file
        self.load_data = load_data  # load data path file
        self.E_max = E_cap  # maximum energy cap of the battery
        self.E_min = (1 - 0.8) * self.E_max  # minimum energy cap of the battery
        self.E_init = 0.3 * self.E_max  # initial starting capacity of the battery
        self.learning_rate = learning_rate  # importance to new and old Q values


def get_default_object():
    '''
        get a default object from here with default variable values
        some examples of price schemes

        #price = [.040,.040,.040,.040,.040,.040,.080,.080,.080,.080,.040,.040,.080,.080,.080,.040,.040,.120,.120,.040,.040,.040,.040,.040]
        #price = [.040,.040,.080,.080,.120,.240,.120,.040,.040,.040,.040,.080,.120,.080,.120,.040,.040,.120,.120,.040,.040,.040,.040,.040]
        #price = [.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.080, .080,.120,.120,.040,.040,.040]
	'''
    gamma = 0.2
    eta = 0.9
    day_chunk = 1
    total_years = 2000
    e_cap = 6.0
    p_cap = 3.0
    e_init = 0.3 * e_cap
    epsilon = 1.0
    actions = np.arange(- p_cap, p_cap + 0.01, 0.5).tolist()
    look_ahead = 1
    solar_data = './Data/solar_double.csv'
    load_data = './Data/load_data_peak6.csv'
    start = 15
    price_scheme = [.040, .040, .040, .040, .040, .040, .080, .080, .080, .080, .040, .040, .080, .080, .080, .040,
                    .040, .120, .120, .040, .040, .040, .040, .040]
    learning_rate = 0.1
    return EnvironmentOptions(eta, gamma, start, day_chunk, total_years, price_scheme, look_ahead, e_cap, p_cap, e_init,
                              epsilon, actions, solar_data, load_data, learning_rate)
