import numpy as np
import matplotlib.pyplot as plt

N = 20

edge = []
for i in range(N):
	if i == 0 or i == N-1:
		for j in range(N):
			edge.append((i,j))
	else:
		edge.append((i,0))
		edge.append((i,N-1))
print(np.array(edge))
np_edge = np.array(edge)

print(np_edge[10])
random_choice = np_edge[10]
print(random_choice[0])

lattice = np.zeros(shape = (20,20))
print(lattice)
print(lattice[random_choice[0], random_choice[1]])
print(lattice[0,10])