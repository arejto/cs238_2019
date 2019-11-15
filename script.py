INPUT_FILE = 'Data.csv'
import pandas
import random
import numpy as np

DEAD_STATE = 3
TRANSPLANT_HEALTHY = 4
STATE_1A = 0
STATE_1B = 1
STATE_2  = 2



def read_data():
	data = pandas.read_csv(INPUT_FILE)
	# print (data)
	return data

def helper(count, state, p_transplant, p_death_wait, output):
	for i in range(count):
		s  = state
		a  = 0 if random.random() < p_transplant else 1
		if a == 0:
			r  = 1e3
			sp = TRANSPLANT_HEALTHY

		else:
			if random.random() < p_death_wait:
				r = -1e6 
				sp = DEAD_STATE
			else:
				r = -5
				sp = state

		f =  open('output.csv', 'a')
		# output.append((s,a,r,sp))
		print (*(s,a,r,sp), sep=',', file=f)



def main():
	data = read_data()
	output = []

	### 1A
	state = STATE_1A
	count = 26397
	p_transplant = 0.669
	p_death_wait = 0.2021403691
	helper(count, state, p_transplant, p_death_wait, output)


	### 1B
	state = STATE_1B
	count = 15075
	p_transplant = 0.653
	p_death_wait = 0.0711593903
	helper(count, state, p_transplant, p_death_wait, output)


	### 2
	state = STATE_2
	count = 8526
	p_transplant = 0.4606666667
	p_death_wait = 0.08639988476
	helper(count, state, p_transplant, p_death_wait, output)

	
if __name__ == '__main__':
	main()