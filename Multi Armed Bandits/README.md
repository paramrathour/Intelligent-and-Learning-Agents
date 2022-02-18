To run `bandit.py` add values for the following parameters in command line.
	
	--instance in, where in is a path to the instance file.
	
	--algorithm al, where al is one of epsilon-greedy-t1, ucb-t1, kl-ucb-t1, thompson-sampling-t1, ucb-t2, alg-t3, alg-t4.
	
	--randomSeed rs, where rs is a non-negative integer.
	
	--epsilon ep, where ep is a number in [0, 1]. For everything except epsilon-greedy, pass 0.02.
	
	--scale c, where c is a positive real number.
	
	--threshold th, where th is a number in [0, 1].
	
	--horizon hz, where hz is a non-negative integer.