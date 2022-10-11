# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import rand
from tqdm import tqdm
from scipy.optimize import curve_fit
from numba import njit

# %%
#input: L = initial lattice
@njit
def Soc(L, counts = 0, xcounts = 0, ycounts = 0):#, N, T):
	N = len(L)
	lattice = np.zeros((N+2, N+2))

	lattice[1:N+1, 1:N+1] = L
	#avalanche = []
	#for t in range(0,T):
	if np.all(lattice < 2 ):
		r1 = np.random.randint(1,N)
		r2 = np.random.randint(1,N)
		lattice[r1,r2]+=1

	else:

		add = np.zeros((N+2, N+2))
		for i in range(1, len(lattice)-1):
			for j in range(1, len(lattice)-1):
				if lattice[i,j]>=2:
					add[i,j] -= 2

					r_neigh = np.random.choice(np.arange(0,4),2)

					while r_neigh[0] == r_neigh[1]:
						r_neigh = np.random.choice(np.arange(0,4),2)
						
					if np.any(r_neigh == 0):
						add[i+1,j] += 1
						ycounts += 1

					if np.any(r_neigh == 1):
						add[i-1, j] += 1
						ycounts -= 1

					if np.any(r_neigh == 2):
						add[i,j+1] += 1
						xcounts += 1

					if np.any(r_neigh == 3):
						add[i, j-1] += 1
						xcounts -= 1

					#add[i+int(np.random.choice(np.array([1,-1]),1)),j] +=1
					#add[i,j+int(np.random.choice(np.array([1,-1]),1))] +=1

					counts +=1

		lattice = lattice + add

		if np.all(lattice < 2 ):
			counts = 0
			xcounts = 0
			ycounts = 0
	#avalanche.append(counts)


	return counts, lattice[1:N+1, 1:N+1], xcounts, ycounts


a1=np.random.randint(2,size = (25,25))
a2=np.random.randint(2,size = (50,50))
a3=np.random.randint(2,size = (100,100))
a4=np.random.randint(2,size = (200,200))

#@njit
def sandpile(initial):
	#a100 = []
	b100 = []
	maxsave =[]
	a=initial
	c = 0
	xdist = 0
	ydist = 0
	for i in tqdm(range(4000000)):
		#a100.append(a)
		c_new,a, xd, yd = Soc(a,c, xdist, ydist)
		if c_new == 0:
			if c != 0:
				b100.append(c)
				maxsave.append(np.max([abs(xdist),abs(ydist)]))
			c = c_new
			xdist = xd
			ydist = yd

		else:
			c = c_new
			xdist = xd
			ydist = yd
	return b100, maxsave

test_count,  test_dist = sandpile(a3)

# %%
plt.hist(test_dist,bins = 50, density = True, color = 'orange')
plt.show()
plt.close('all')

d = np.log(test_dist)/np.log(100)

def dimension(L,d):
	return L**d

plt.scatter(test_dist, test_count, s = 3)
plt.xlabel('Largest linear dimenstion')
plt.ylabel('Avalanche size')
plt.title('Question 3')
#plt.xscale('log')
plt.show()

# %%
print(np.log(np.max(test_dist))/np.log(100))
# %%
