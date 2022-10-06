
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


	return counts, lattice[1:N+1, 1:N+1]


a1=np.random.randint(2,size = (25,25))
a2=np.random.randint(2,size = (50,50))
a3=np.random.randint(2,size = (100,100))
a4=np.random.randint(2,size = (200,200))

#@njit
def sandpile(initial):
	#a100 = []
	b100 = []
	a=initial
	c = 0
	for i in tqdm(range(400000)):
		#a100.append(a)
		c_new,a = Soc(a,c)
		if c_new == 0:
			if c != 0:
				b100.append(c)
			c = c_new

		else:
			c = c_new
	return b100

#print(sandpile(a1))
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

sp_25 = sandpile(a1)
sp_50 = sandpile(a2)
sp_100 = sandpile(a3)
sp_200 = sandpile(a4)

plt.hist(sp_25, density = True)
plt.show()
plt.close('All')



y0, x0 = np.histogram(sp_25, bins = np.logspace(start = np.log(1), stop = np.log(np.max(sp_25))),  density = True)
y1, x1 = np.histogram(sp_50, bins = np.logspace(start = np.log(1), stop = np.log(np.max(sp_50))),  density = True)
y2, x2 = np.histogram(sp_100, bins = np.logspace(start = np.log(1), stop = np.log(np.max(sp_100))),  density = True)
y3, x3 = np.histogram(sp_200, bins = np.logspace(start = np.log(1), stop = np.log(np.max(sp_200))),  density = True)

plt.plot(x0[1:50], y0, 'o', label = 'N = 25')
plt.plot(x1[1:50], y1, 'o', label = 'N = 50')
plt.plot(x2[1:50], y2, 'o', label = 'N = 100')
plt.plot(x3[1:50], y3, 'o', label = 'N = 200')
plt.xscale('log')
plt.yscale('log')
plt.title('Powerlaws for diff. system sizes')
plt.xlabel('log(s)')
plt.ylabel('log(P(s))')
plt.xlim(0, 10**6)
plt.legend()
plt.show()
plt.close('All')

#fig, axs = plt.subplots(2,2)
#axs[0,0].plot(x0[1:150],y0)
#axs[0,1].plot(x1[1:150],y1)
#axs[1,0].plot(x2[1:150],y2)
#axs[1,1].plot(x3[1:150],y3)
#plt.xscale('log')
#plt.show()



t = np.linspace(0,200000,len(sp_25))
print(len(t), len(sp_25))
plt.plot(t, sp_25)
plt.show()
plt.close('All')


