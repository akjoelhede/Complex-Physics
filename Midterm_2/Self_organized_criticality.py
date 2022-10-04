import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import rand
from tqdm import tqdm
from scipy.optimize import curve_fit
from numba import njit


#input: L = initial lattice
@njit
def Soc(L, counts = 0):#, N, T):
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

					if np.any(r_neigh == 1):
						add[i-1, j] += 1

					if np.any(r_neigh == 2):
						add[i,j+1] += 1

					if np.any(r_neigh == 3):
						add[i, j-1] += 1

					#add[i+int(np.random.choice(np.array([1,-1]),1)),j] +=1
					#add[i,j+int(np.random.choice(np.array([1,-1]),1))] +=1

					counts +=1

		lattice = lattice + add

		if np.all(lattice < 2 ):
			counts = 0
	#avalanche.append(counts)


	return lattice[1:N+1, 1:N+1], counts 


#a1=np.random.randint(2,size = (25,25))
a2=np.random.randint(2,size = (50,50))
#a3=np.random.randint(2,size = (100,100))
#@njit
def sandpile(initial):
	a100 = []
	b100 = []
	a=initial
	c = 0
	for i in tqdm(range(200000)):
		a100.append(a)
		a,c_new = Soc(a,c)
		if c_new == 0:
			if c != 0:
				b100.append(c)
			c = c_new

		else:
			c = c_new

	return b100

#anim_50 = sandpile(a2)[0]
#print(a100[1].sum(), a100[999].sum())

#fig = plt.figure(figsize = (8,8))
#im = plt.imshow(anim_50)
#plt.colorbar()
#def animate_func(i):
	#im.set_array(anim_50[i])
	#return [im]

#anim = animation.FuncAnimation(fig, animate_func, frames = 1000, interval = 20)
#plt.show()
#plt.close('All')

#sp_25 = sandpile(a1)
sp_50 = sandpile(a2)
#sp_100 = sandpile(a3)

plt.hist(sp_50, density = True)
plt.show()
plt.close('All')


y, x = np.histogram(sp_50, bins = np.logspace(start = np.log(1), stop = np.log(np.max(sp_50))),  density = True)
plt.plot(x[1:150], y, 'o')
plt.xscale('log')
plt.xlim(0, 10**6)

plt.yscale('log')
plt.show()
plt.close('All')


t = np.linspace(0,150000,len(sp_50))
print(len(t), len(sp_50))
plt.plot(t, sp_50)
plt.show()


