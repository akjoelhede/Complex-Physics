import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


#input: L = initial lattice
def Soc(L):#, N, T):
	N = len(L)
	lattice = np.zeros((N+2, N+2))
	#lattice[1:N+1, 1:N+1] = np.random.randint(5,size = (L,L))
	lattice[1:N+1, 1:N+1] = L

	#for t in range(0,T):
	if np.any(4 < lattice[1:N+1, 1:N+1]):
		r1 = np.random.randint(1,N)
		r2 = np.random.randint(1,N)
		lattice[r1,r2]+=1

	else:
		add = np.zeros((N+2, N+2))

		for i in range(1, len(lattice)-1):
			for j in range(1, len(lattice)-1):
				if lattice[i,j]>=4:
					add[i,j] -= -4
					add[i+1,j] += 1
					add[i,j+1] += 1
					add[i-1,j] += 1
					add[i,j-1] += 1
		lattice = lattice+add
	return lattice[1:N+1, 1:N+1]

a100 = []
a=np.random.randint(4,size = (20,20))
for i in range(1000):
	a100.append(a)
	a = Soc(a)

print(a100[1].sum(), a100[999].sum())

img = plt.imshow(a100[999])
plt.colorbar(img)
plt.show()

