import numpy as np

class EnvironmentOptions :

	def __init__(self, eta, gamma, start, day_chunk, total_years, price_scheme, look_ahead, E_cap, P_cap, E_init, epsilon, \
				 actions, solar_data, load_data) :
		'''
			options class from where environment get's all it's parameters.
			can be custom created or a default veriosn can also be loaded.
			added to provide flexibility as well as make the process of creating
			the environment uniform, like other openai-gym environments
		'''
		self.eta = eta
		self.gamma = gamma
		self.start = start
		self.day_chunk = day_chunk
		self.total_years = total_years
		self.price_scheme = price_scheme
		self.E_cap = E_cap
		self.P_cap = P_cap
		self.E_init = E_init
		self.epsilon = epsilon
		self.actions = actions
		self.look_ahead = look_ahead
		self.solar_data = solar_data
		self.load_data = load_data



def getDefaultObject() :
	'''
		get a default object from here with default variable values
		some examples of price schemes

		#price = [.040,.040,.040,.040,.040,.040,.080,.080,.080,.080,.040,.040,.080,.080,.080,.040,.040,.120,.120,.040,.040,.040,.040,.040]
		#price = [.040,.040,.080,.080,.120,.240,.120,.040,.040,.040,.040,.080,.120,.080,.120,.040,.040,.120,.120,.040,.040,.040,.040,.040]
		#price = [.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.040,.080, .080,.120,.120,.040,.040,.040]
	'''
	gamma = 0.99
	eta = 0.9
	day_chunk = 10
	total_years = 2000
	E_cap = 6.0
	P_cap = 3.0
	E_init = 0.3*E_cap
	epsilon = 1.0
	actions = np.arange(- P_cap, P_cap + 0.01, 0.5).tolist()
	total_number_hours = 24
	look_ahead = 1
	solar_data = './Data/solar_double.csv'
	load_data = './Data/load_data_peak6.csv'
	start = 0
	price_scheme = [.040,.040,.040,.040,.040,.040,.080,.080,.080,.080,.040,.040,.080,.080,.080,.040,.040,.120,.120,.040,.040,.040,.040,.040]
	return EnvironmentOptions(eta, gamma, start, day_chunk, total_years, price_scheme, look_ahead, E_cap, P_cap, E_init, epsilon, \
				 actions, solar_data, load_data) #please fill with default values from main class
