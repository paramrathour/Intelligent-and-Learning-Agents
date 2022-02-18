import argparse as ap
import numpy as np
import math
import matplotlib.pyplot as plt

algorithms = ["epsilon-greedy-t1", "ucb-t1", "kl-ucb-t1", "thompson-sampling-t1", "ucb-t2", "alg-t3", "alg-t4"]
parser = ap.ArgumentParser()
parser.add_argument("--instance")
parser.add_argument("--algorithm")
parser.add_argument("--randomSeed")
parser.add_argument("--epsilon")
parser.add_argument("--scale")
parser.add_argument("--threshold")
parser.add_argument("--horizon")

arguments = parser.parse_args()

ins = arguments.instance
al = arguments.algorithm
rs = int(arguments.randomSeed)
ep = float(arguments.epsilon)
c = float(arguments.scale)
th = float(arguments.threshold)
hz = int(arguments.horizon)
probabilities = []
rewards = np.array([1])
HIGHS = 0

algo = algorithms.index(al)

instanceFile = open(ins, "r")

if algo < 5:
	for lines in instanceFile:
		pArm = float(lines.rstrip())
		probabilities.append(pArm)
	numberOfArms = len(probabilities)
	bestExpectedReward = max(probabilities)
else:
	rewards = np.array(list(map(float,(instanceFile.readline()).split())))
	lines = instanceFile.readlines()
	for line in lines:
		probs = np.array(list(map(float,line.split())))
		probabilities.append(probs)
	numberOfArms = len(probabilities)
	expectedRewards = [np.dot(rewards, probabilities[i]) for i in range(numberOfArms)]
	# print(expectedRewards)
	bestExpectedReward = max(expectedRewards)

instanceFile.close()

def bernoulliSamples(th, probabilities, armChosen, currentReward, rewards):
	return np.random.binomial(1, probabilities[armChosen])

def weightedSamples(th, probabilities, armChosen, currentReward, rewards):
	return np.random.choice(rewards, p = probabilities[armChosen])

def weightedSampleswithThreshold(th, probabilities, armChosen, currentReward, rewards):
	return 1 if (np.random.choice(rewards, p = probabilities[armChosen]) > th) else 0

def epsilonGreedyChoosingArm(ep, c, t, armData, numberOfArms, rewards):
	if np.random.random() < ep:
		return np.random.randint(0, numberOfArms)
	else:
		pat = [armData[arm][1] / armData[arm][0] for arm in range(numberOfArms)]
		return np.argmax(pat)

def UCBChoosingArm(ep, c, t, armData, numberOfArms, rewards):
	return np.argmax([(armData[arm][1] / armData[arm][0]) + math.sqrt(c*math.log(t)/ armData[arm][0]) for arm in range(numberOfArms)])

def KL(p,q):
	if p == 0:
		return 0
	elif q == 0 or p == 1 or q == 1:
		return float('inf')
	else:
		return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))

def KLmaxUCB(c,t):
	return math.log(t) + c*math.log(math.log(t))

def KLupperConfidenceBound(p, u, c, t):
	maxBound = KLmaxUCB(c,t)/u
	q = p
	b = (1 - p) / 2
	while b > 1e-4:
		if KL(p,q+b) <= maxBound:
			q += b
		b /= 2;
	return q

def KLUCBChoosingArm(ep, c, t, armData, numberOfArms, rewards):
	return np.argmax([KLupperConfidenceBound(armData[arm][1] / armData[arm][0], armData[arm][0], 3, t) for arm in range(numberOfArms)])

def thompsonChoosingArm(ep, c, t, armData, numberOfArms, rewards):
	return np.argmax([np.random.beta(armData[arm][1] + 1, armData[arm][0] - armData[arm][1] + 1) for arm in range(numberOfArms)])

def thompsonChoosingArmGeneralized(ep, c, t, armData, numberOfArms, rewards):
	generalizedSamples = [np.random.dirichlet(armData[arm][1:]+1) for arm in range(numberOfArms)]
	return np.argmax([np.dot(generalizedSamples[arm], rewards) for arm in range(numberOfArms)])

def updateRecords(th, probabilities, armData, armChosen, currentReward, algo, rewards):
	armData[armChosen][0] += 1
	sample = algoToSamplingMapping[al](th, probabilities, armChosen, currentReward, rewards)
	if algo == 5:
		armData[armChosen][1 + int(np.argwhere(rewards == sample))] += 1
	else:
		armData[armChosen][1] += sample
	return currentReward + sample

def chooseArm(ep, c, t, armData, numberOfArms, rewards):
	return algoToChoosingArmMapping[al](ep, c, t, armData, numberOfArms, rewards)
	
def task(probabilities, rs, ep, c, th, hz):
	currentReward = 0
	armData = np.zeros((numberOfArms,1 + len(rewards)), dtype = int)
	np.random.seed(rs)

	for arm in range(min(numberOfArms, hz)):
		currentReward = updateRecords(th, probabilities, armData, arm, currentReward, algo, rewards)

	for i in range(numberOfArms, hz):
		armChosen = chooseArm(ep, c, i, armData, numberOfArms, rewards)
		currentReward = updateRecords(th, probabilities, armData, armChosen, currentReward, algo, rewards)
	# print(armData)
	return currentReward

algoToSamplingMapping = {"epsilon-greedy-t1" : bernoulliSamples, "ucb-t1" : bernoulliSamples, "kl-ucb-t1" : bernoulliSamples, "thompson-sampling-t1" : bernoulliSamples, "ucb-t2" : bernoulliSamples, "alg-t3" : weightedSamples, "alg-t4" : weightedSampleswithThreshold}

algoToChoosingArmMapping = {"epsilon-greedy-t1" : epsilonGreedyChoosingArm, "ucb-t1" : UCBChoosingArm, "kl-ucb-t1" : KLUCBChoosingArm, "thompson-sampling-t1" : thompsonChoosingArm, "ucb-t2" : UCBChoosingArm, "alg-t3" : thompsonChoosingArmGeneralized, "alg-t4" : thompsonChoosingArm}

reward = task(probabilities, rs, ep, c, th, hz)
# print(probabilities)
# print(bestExpectedReward)

if al == 'alg-t4':
	HIGHS = reward
	indices  = np.argwhere(rewards > th)
	# print(indices)
	bestExpectedReward = max([sum([probabilities[arm][i] for i in indices]) for arm in range(numberOfArms)])

REG = float(bestExpectedReward * hz - reward)

output = ins + ", " + al + ", " + "%g" % (rs) + ", " + "%g" % round(ep, 3) + ", " + "%g" % round(c, 3) + ", " + "%g" % round(th, 3) + ", " + "%g" % (hz) + ", " + "%g" % round(REG, 4) + ", " + "%g" % (HIGHS)# + "\n"
print(output)