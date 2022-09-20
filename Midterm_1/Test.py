import numpy as np

def ran_neigh(M):
	Random_neighbours = []
	for i in range(len(M)):


		if i == 0:
			r_n = i + 1
			l_n = len(M)-1
		
		if i == len(M)-1:
			r_n = 0
			l_n = i-1

		else:
			r_n = i+1
			l_n = i-1


		rand_neig1 = np.random.choice(len(M))
		while rand_neig1 == i or rand_neig1 == l_n or rand_neig1 == r_n:
			rand_neig1=np.random.choice(len(M))

		rand_neig2 = np.random.choice(len(M))
		while rand_neig2 == i or rand_neig2 == l_n or rand_neig2 == r_n or rand_neig2 == rand_neig1:
			rand_neig2 = np.random.choice(len(M))

		Random_neighbours.append(np.array([rand_neig1, rand_neig2]))


	return Random_neighbours

test_array = np.arange(1,20)


print(ran_neigh(test_array))