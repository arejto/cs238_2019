INPUT_FILE = 'Data.csv'
import pandas
import random
import numpy as np

# constants representing the status of the patient
STATE_1A = 0
STATE_1B = 1
STATE_2  = 2
DEAD_STATE = 3
TRANSPLANT_HEALTHY = 4

# rewards associated with each action
WAIT_REWARD = -5
DEATH_REWARD = -500

NUM_SAMPLES = 50000
STARTING_NUM_HEARTS = NUM_SAMPLES / 10 

# the transition probabilities for each state
state_prob_dict = {STATE_1A : {'p_state': 0.52794, 'p_transplant': 0.669, 'p_death_wait': 0.2021403691},
				   STATE_1B : {'p_state': 0.3015, 'p_transplant': 0.653, 'p_death_wait': 0.0711593903},
				   STATE_2 : {'p_state': 0.17052, 'p_transplant': 0.4606666667, 'p_death_wait': 0.08639988476}}

def read_data():
	data = pandas.read_csv(INPUT_FILE)
	# print (data)
	return data

''' Calculates the opportunity cost for a heart transplant. This is a heuristic, linear function that
decreases the reward of transplanting a heart as the supply of hearts goes down
'''
def opportunity_cost(num_hearts):
	return .1 * (STARTING_NUM_HEARTS - (1.25 * (STARTING_NUM_HEARTS - num_hearts)))

def allocate_hearts(num_hearts, transplant_reward, f):
	rand = random.random()
	s  = STATE_1A if rand > state_prob_dict[STATE_1A]['p_state'] else STATE_1B if rand > state_prob_dict[STATE_1B]['p_state'] else STATE_2
	a  = 0 if random.random() < state_prob_dict[s]['p_transplant'] else 1
	if a == 0:
		r  = transplant_reward
		sp = TRANSPLANT_HEALTHY
		num_hearts -= 1
		transplant_reward = opportunity_cost(num_hearts)
	else:
		if random.random() < state_prob_dict[s]['p_death_wait']:
			r =  DEATH_REWARD
			sp = DEAD_STATE
		else:
			r = WAIT_REWARD
			sp = s

	print (*(s,a,r,sp), sep=',', file=f)
	return num_hearts, transplant_reward


def main():
	num_hearts = STARTING_NUM_HEARTS
	# reward for transplanting will change as the number of hearts available decreases
	transplant_reward = STARTING_NUM_HEARTS / 10

	f =  open('output.csv', 'a')
	for i in range(NUM_SAMPLES):
		num_hearts, transplant_reward = allocate_hearts(num_hearts, transplant_reward, f)
	f.close()

if __name__ == '__main__':
	main()