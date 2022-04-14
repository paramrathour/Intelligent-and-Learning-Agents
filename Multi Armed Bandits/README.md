![](multi-armed-bandit.gif)
## Introduction 
In a bandit instance, each arm provides a random reward from a probability distribution specific to that arm, this distribution is not known a-priori and it may even change. The objective of the gambler is to maximize the sum of rewards earned through the arms which is same as minimising the 'regret'.

## Tasks Implementation
- Different algorithms for sampling the arms and regret-minimisation of a Bernoulli multi-armed bandit using ε-greedy exploration, UCB, KL-UCB, and Thompson Sampling 
- Optimised the scaling coefficient of the exploration bonus in UCB 
- An efficient a regret-minimisation algorithm for discrete non Bernoulli bandit instances

## Other Details
To run `bandit.py` add values for the following parameters in command line.
	
	--instance in, where in is a path to the instance file.
	
	--algorithm al, where al is one of epsilon-greedy-t1, ucb-t1, kl-ucb-t1, thompson-sampling-t1, ucb-t2, alg-t3, alg-t4.
	
	--randomSeed rs, where rs is a non-negative integer.
	
	--epsilon ep, where ep is a number in [0, 1]. For everything except epsilon-greedy, pass 0.02.
	
	--scale c, where c is a positive real number.
	
	--threshold th, where th is a number in [0, 1].
	
	--horizon hz, where hz is a non-negative integer.