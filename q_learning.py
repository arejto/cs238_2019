import random
import pandas
import numpy as np
import collections
import time

INPUT_FILE  = 'outputv2.csv'
OUTPUT_FILE = 'bogusv2.policy'
ALL_STATES  = set(range(np.prod([10, 100, 5])))
state_action_map = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
S = set([])
A = set([])
lam = 1
not_seen = None


def create_state_action_map(data):
	global state_action_map
	global S
	global A
	global not_seen
	for s, a, r, sp in data:
		S.add(s)
		A.add(a)
		state_action_map[s][a].append((r, sp))

	not_seen = ALL_STATES-S
	return state_action_map


# def bellman_equation(s, a):
#     global state_action_map
#     global lam
#     r = state_action_map[s][a][0]
#     next_state = state_action_map[s][a][1]
#     possible_actions = list(state_action_map[next_state].keys())
#     return r + lam * max([bellman_equation(next_state, poss_action)
#                             for poss_action in possible_actions])
#
# def q_learning():
#     global state_action_map
#     global lam
#     q = collections.defaultdict()
#     for time in range(10):
#         for state in state_action_map.keys():
#             for action in state_action_map[state].keys():
#                 new_q[(state, action)] =
#
#     return q


# def ValueIteration():
#     global state_action_map
#     global lam
#     k = 0
#     U      = None
#     next_U = collections.defaultdict(lambda: 0)
#     states = state_action_map.keys()

#     while not U == next_U:
#         U = next_U.copy()
#         for s in states:
#             possible_actions = list(state_action_map[s].keys())
#             next_U[s-1] = max([state_action_map[s][a][0] + lam * U[state_action_map[s][a][1] - 1] for a in possible_actions])
#         k += 1

#     return next_U


def getPolicy(Q):
    global state_action_map
    global lam
    global S
    global A
    global ALL_STATES
    pi = np.zeros(len(ALL_STATES)+1)

    for s in S:
        possible_actions = list(A)
        ## index into possible actions
        pi[s] = int(possible_actions[
                            np.argmax([Q[s,a] for a in possible_actions])
                      ])
        # print (Q[s], pi[s])
    return pi

def SarsaLambdaLearning(data):
	global state_action_map
	global S
	global A
	global lam
	global not_seen
	global ALL_STATES
	gamma = 0.95
	# alpha = 0.99
	Q = np.zeros((len(ALL_STATES)+1, len(A)+1))
	N = collections.defaultdict(int)
	# st, at = data[0][0], data[0][1]
	# policy = getPolicy(Q)
	# st, at = data[0][0], data[0][1]
	t = 0
	visited_s = set([])
	Q_old = None
	for epoch in range(100):
		print (epoch)
		if np.all(Q_old == Q):
			break
		Q_old = Q.copy()

		for i in range(len(data)-2):
			# possible_r_sps = state_action_map[st][at]

			# r, st_1 = possible_r_sps[int(len(possible_r_sps) * random.random())]
			# if t < 10000:
			# 	if not len(list(state_action_map[st_1].keys())):
			# 		break
			# 	at_1 = random.choice(list(state_action_map[st_1].keys()))

			# else:
			# 	if t % 10000:
			# 		policy = getPolicy(Q)
			# 	at_1 = policy[st_1]
			
			# ending spot
			st,at,rt,spt = data[i]
			st_1,at_1,rt_1,spt_1 = data[i+1]
			
			if spt != st_1:
				continue

			N[(st,at)] += 1
			alpha = 1 / len(data)
			# print (st)
			maxQ_overa = max(Q_old[st_1,:])
			# print ('here')
			# print (Q[st_1,at])
			# print (maxQ_overa)
			# print (maxQ_overa)
			# print (alpha *(rt + gamma*maxQ_overa - Q[st,at]))

			Q[st,at] = Q_old[st,at] + alpha *(rt + (gamma*maxQ_overa) - Q_old[st,at])  ### alpha = 1/N if N != 0 else 1
			# print (Q[st,at])
			# squig = rt + gamma * Q[(st_1, at_1)] - Q[(st, at)]
			# for s in visited_s:
			# 	for a in A:
			# 		Q[(s,a)] = Q[(s,a)] + alpha * squig * N[(s,a)]  ### alpha = 1/N if N != 0 else 1
			# 		N[(s,a)] = gamma * lam * N[(s,a)]
			
			# visited_s.add(st)
			t += 1

	return Q

def handleUnVisited(policy):
	LOCAL = True
	i = 0
	if LOCAL:
		for new_state in not_seen:
			i += 1
			# print (new_state)
			if i % 100 == 1:
				closest_seen = min(S, key= lambda x:abs(x - new_state))
			policy[new_state] = policy[closest_seen]
	else:
		pass

def generateOutput(optimal_pi):
	with open(OUTPUT_FILE, 'w+') as f:
		for action in optimal_pi[1:]:
			f.write('%d\n' % action)


def main():
    global state_action_map
    start_time = time.time()
    data = np.array(pandas.read_csv(INPUT_FILE, dtype=int))
    state_action_map = create_state_action_map(data)
    Q = SarsaLambdaLearning(data)
    optimal_policy = getPolicy(Q)
    print ('hi alex')
    handleUnVisited(optimal_policy)
    print (len(optimal_policy))
    print ('hi alex')
    generateOutput(optimal_policy)
    print ('total_time: ', time.time()-start_time)
if __name__ == '__main__':
    main()
