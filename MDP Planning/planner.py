import argparse as ap
import numpy as np
import math
import pulp

algorithms = ["vi", "hpi", "lp"]
parser = ap.ArgumentParser()
parser.add_argument("--mdp")
parser.add_argument("--algorithm", default = algorithms[0])

arguments = parser.parse_args()
mdpPath = arguments.mdp
al = arguments.algorithm

mdpFile = open(mdpPath, "r")
numStates = int(mdpFile.readline().split()[1])
numActions = int(mdpFile.readline().split()[1])
endStates = np.zeros(numStates, dtype = bool)
for state in mdpFile.readline().split()[1:]:
	if int(state) != -1:
		endStates[int(state)] = True
# endStates = [int(state) for state in mdpFile.readline().split()[1:]]
transitions = [[[] for action in range(numActions)] for state in range(numStates)]
while True:
	line = mdpFile.readline().split()
	if line[0] == 'transition':
		s1 = int(line[1])
		ac = int(line[2])
		s2 = int(line[3])
		r  = float(line[4])
		p  = float(line[5])
		transitions[s1][ac].append([s2, r, p])
	else:
		break
mdpType = line[1]
discount = float(mdpFile.readline().split()[1])
# print(numStates,numActions,endStates,transitions,mdpType,discount)
mdpFile.close()

def valueIteration(numStates, numActions, endStates, transitions, mdpType, discount):
	values = np.zeros(numStates)
	newValues = np.zeros(numStates)
	actions = np.zeros(numStates, dtype = int)
	while True:
	    for s1 in range(numStates):
	        if not endStates[s1]:
	            temp = np.zeros(numActions)
	            for ac in range(numActions):
	                try:
	                	data = transitions[s1][ac]
	                except:
	                	continue
	                temp[ac] = sum([data[i][2]*(data[i][1]+discount*values[data[i][0]]) for i in range(len(data))])
	            actions[s1] = np.argmax(temp)
	            newValues[s1] = temp[actions[s1]]
	    if np.linalg.norm(values - newValues, ord = np.inf) < 1e-10:
	    	break
	    else:
	        values = [newValues[i] for i in range(numStates)]
	values = [newValues[i] for i in range(numStates)]
	return values, actions

def howardPolicyIteration(numStates, numActions, endStates, transitions, mdpType, discount):
	actions = np.zeros(numStates, dtype = int)
	improvableStates = True
	while improvableStates:
		improvableStates = False
		values = np.zeros(numStates)
		newValues = np.zeros(numStates)
		while True:
			for s1 in range(numStates):
				if not endStates[s1]:
					data = transitions[s1][actions[s1]]
					newValues[s1] = sum([data[i][2]*(data[i][1]+discount*values[data[i][0]]) for i in range(len(data))])
			if np.linalg.norm(values - newValues, ord = np.inf) < 1e-10:
				break
			else:
				values = [newValues[i] for i in range(numStates)]
		for s1 in range(numStates):
			for ac in range(numActions):
				data = transitions[s1][ac]
				actionValues = sum([data[i][2]*(data[i][1]+discount*values[data[i][0]]) for i in range(len(data))])
				if actionValues > values[s1] and (actionValues - values[s1] > 1e-7):
					improvableStates = True
					values[s1] = actionValues
					actions[s1] = ac
					break
		# print(values, actions)

	return values, actions

def linearProgramming(numStates, numActions, endStates, transitions, mdpType, discount):
	problem = pulp.LpProblem("Values", pulp.LpMinimize)
	decisionVariables = [pulp.LpVariable('V('+ str(state) +')') for state in range(numStates)]
	cost = sum(decisionVariables)
	problem += cost

	for s1 in range(numStates):
		for ac in range(numActions):
			data = transitions[s1][ac]
			problem += (decisionVariables[s1] >= pulp.lpSum([data[i][2]*(data[i][1]+discount*decisionVariables[data[i][0]]) for i in range(len(data))]))

	# problem.writeLP("LinearProgrammingFormuation.lp")
	# print(problem)
	problem.solve(pulp.PULP_CBC_CMD(msg = None))
	values = np.zeros(numStates)
	for value in problem.variables():
		values[int(value.name[2:-1])] = value.varValue
	actions = np.zeros(numStates, dtype = int)
	for s1 in range(numStates):
		actionValues = np.zeros(numActions, dtype = int)
		for ac in range(numActions):
			data = transitions[s1][ac]
			actionValues[ac] = sum([data[i][2]*(data[i][1]+discount*values[data[i][0]]) for i in range(len(data))])
		actions[s1] = np.argmin([abs(values[s1] - i) for i in actionValues])
	return values, actions

algoToFunctionMapping = {"vi" : valueIteration, "hpi" : howardPolicyIteration, "lp" : linearProgramming}
# algoToFunctionMapping = {"vi" : valueIteration, "hpi" : howardPolicyIteration, "lp" : valueIteration}

values, actions = algoToFunctionMapping[al](numStates, numActions, endStates, transitions, mdpType, discount)
output = ""
for state in range(numStates):
	output += str(round(values[state], 7)) + "\t" + str(round(actions[state], 7)) + "\n"
print(output)