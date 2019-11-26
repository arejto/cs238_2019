import pandas
import random
import numpy as np
from scipy.stats import multivariate_normal

INPUT_FILE = 'Data.csv'
MAX_HEARTS        = 10
MAX_WEEKS_ON_LIST = 100
MAX_HEALTHY_LEVEL = 5


# constants representing the status of the patient
STATE_1A = 0
STATE_1B = 1
STATE_2  = 2
DEAD_STATE = 3
TRANSPLANT_HEALTHY = 4

# rewards associated with each action
WAIT_REWARD  = 30
DEATH_REWARD = -500

NUM_SAMPLES = 50000
STARTING_NUM_HEARTS = MAX_HEARTS - 1


# true probs I guess
# the transition probabilities for each state
state_prob_dict = {STATE_1A : {'p_state': 0.52794, 'p_transplant': 0.669, 'p_death_wait': 0.2021403691},
				   STATE_1B : {'p_state': 0.3015, 'p_transplant': 0.653, 'p_death_wait': 0.0711593903},
				   STATE_2 : {'p_state': 0.17052, 'p_transplant': 0.4606666667, 'p_death_wait': 0.08639988476}}


# from this table: Organ Procurement and Transplantation Network Competing Risk Median Waiting Time to Deceased Donor Transplant For Registrations Listed : 2003-2014
wait_time_dict = { STATE_1A : {'mean': 87,  'cov': 14/2},
				   STATE_1B : {'mean': 253, 'cov': (273 - 240)/2},
				   STATE_2  : {'mean': 726, 'cov': (817 - 657)/2 }
			     }


def read_data():
	data = pandas.read_csv(INPUT_FILE)
	# print (data)
	return data


''' Calculates the opportunity cost for a heart transplant. This is a heuristic, linear function that
decreases the reward of transplanting a heart as the supply of hearts goes down
'''
def opportunity_cost(num_hearts):
	return .1 * (STARTING_NUM_HEARTS - (1.25 * (STARTING_NUM_HEARTS - num_hearts)))


def get_state_num(health, num_hearts, wait_time):
	return np.ravel_multi_index((health, num_hearts, wait_time), (MAX_HEALTHY_LEVEL, MAX_HEARTS, MAX_WEEKS_ON_LIST))


'''
	Heuristic for reward based on real physician action.
	higher reward for transplanting near mean. Slight negative reward 
	for transplanting early or leate
'''
def reward_by_wait_time(health, wait_time):
	# convert weeks to days
	wait_time *= 7
	reward = multivariate_normal.pdf(wait_time, mean=wait_time_dict[health]['mean'], cov=wait_time_dict[health]['cov'])
	return reward * 10 - 6


def allocate_hearts(health, num_hearts, wait_time, f):
	collapsed_s  = get_state_num(health, num_hearts, wait_time)
	a  = 0 if random.random() < state_prob_dict[health]['p_transplant'] else 1
	active_patient = True

	if a == 0:
		r  = reward_by_wait_time(health, wait_time) - opportunity_cost(num_hearts)
		health = TRANSPLANT_HEALTHY
		num_hearts -= 1
		active_patient = False
	
	else:
		if random.random() < state_prob_dict[health]['p_death_wait']:
			r =  DEATH_REWARD
			health = DEAD_STATE
			active_patient = False
		else:
			r = WAIT_REWARD
			wait_time += 1

	collapsed_sp = get_state_num(health, num_hearts, wait_time)
	print (*(collapsed_s, a, int(r), collapsed_sp), sep=',', file=f)
	return active_patient, num_hearts, wait_time


def main():
	num_hearts = STARTING_NUM_HEARTS
	# reward for transplanting will change as the number of hearts available decreases
	# not sure about this !!! TODO:
	# transplant_reward = STARTING_NUM_HEARTS / 10

	f =  open('outputv2.csv', 'a')
	for i in range(NUM_SAMPLES):
		if num_hearts < 1:
			num_hearts = STARTING_NUM_HEARTS

		rand = random.random()
		wait_time = 0
		health  = STATE_1A if rand > state_prob_dict[STATE_1A]['p_state'] else STATE_1B if rand > state_prob_dict[STATE_1B]['p_state'] else STATE_2
		active_patient = True
		while active_patient:
			active_patient, num_hearts, wait_time = allocate_hearts(health, num_hearts, wait_time, f)
	f.close()

if __name__ == '__main__':
	main()