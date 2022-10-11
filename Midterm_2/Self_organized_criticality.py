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
	for i in tqdm(range(4000000)):
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

# %%

#*Specifies the lattice sizes used 
sp_25 = sandpile(a1)
sp_50 = sandpile(a2)
sp_100 = sandpile(a3)
sp_200 = sandpile(a4)

# %%
#*Calculates the transient for the smallest system of 25x25 lattice
t = np.linspace(0,1000000,len(sp_25))
print(len(t), len(sp_25))
plt.plot(t, sp_25)
plt.xscale('log')
plt.xlabel('t')
plt.ylabel('s (size of avalanches)')
plt.title('Sequence of avalaches as a funciton of time')
plt.show()
plt.close('All')

#* Shows a histogram of the distribution of the sizes of avalanches for all system sizes
#plt.hist(sp_200, density= True)
#plt.show()
#plt.close('all')

#fig = plt.figure(figsize=(8,6))
#fig, axs = plt.subplots(2,2)
#axs[0,0].hist(sp_25, density = True)
#axs[0,1].hist(sp_50, density = True)
#axs[1,0].hist(sp_100, density = True)
#axs[1,1].hist(sp_200, density = True)
#plt.xscale('log')
#plt.show()
#plt.close('all')


#*Extracts probabilities and sizes of avalanches from a histogram
y0, x0 = np.histogram(sp_25, bins = np.logspace(start = np.log(1), stop = np.log(np.max(sp_25))),  density = True)
y1, x1 = np.histogram(sp_50, bins = np.logspace(start = np.log(1), stop = np.log(np.max(sp_50))),  density = True)
y2, x2 = np.histogram(sp_100, bins = np.logspace(start = np.log(1), stop = np.log(np.max(sp_100))),  density = True)
y3, x3 = np.histogram(sp_200, bins = np.logspace(start = np.log(1), stop = np.log(np.max(sp_200))),  density = True)

#todo: ur girlfriend
#*finds bin centers from above histograms
x0centers = (x0[:-1] + x0[1:])/2
x1centers = (x1[:-1] + x1[1:])/2
x2centers = (x2[:-1] + x2[1:])/2 
x3centers = (x3[:-1] + x3[1:])/2

#*Plots the probabilities and sizes in a log/log plot to show the powerlaw structure 

def P1(s,t,D):
	y = 1/(s**t)*np.exp(-s/(50**D))
	return y

#par, cov = curve_fit(P1, x1centers, y1, p0 = [1.5,5.5])
#print("Det her er parametrene, og cov matricens diagonal", par, np.sqrt(np.diag(cov)))

#def Dim(x, L):
	#return np.log(np.max(x))/np.log(L)

#d_25 = Dim(x0centers, 25)
#d_50 = Dim(x1centers, 50)
#d_100 = Dim(x2centers, 100)
#d_200 = Dim(x3centers, 200)

#print(f'The dimension D for the different system are as follows, D_25 = {d_25}, D_50 = {d_50}, D_100 = {d_100}, D_200 = {d_200}')
# %%

plt.plot(x0centers, y0, 'o', label = 'N = 25')
#plt.plot(x0centers, P1(x0centers, *par))
plt.plot(x1centers, y1, 'o', label = 'N = 50')
#plt.plot(x1centers, P1(x1centers, *par))
plt.plot(x2centers, y2, 'o', label = 'N = 100')
plt.plot(x3centers, y3, 'o', label = 'N = 200')
plt.xscale('log')
plt.yscale('log')
plt.title('Powerlaws for diff. system sizes')
plt.xlabel('log(s)')
plt.ylabel('log(P(s))')
plt.xlim(0, 10**6)
plt.legend()
plt.show()
plt.close('All')

tau = 1.3
d = 2.86
x = np.linspace(0,len(x0centers))
#*Extracting tau and D
plt.plot(x0centers/25**d,y0*(x0centers**tau),'o', label = 'N = 25')
plt.plot(x1centers/50**d,y1*(x1centers**tau),'o', label = 'N = 50')
plt.plot(x2centers/100**d,y2*(x2centers**tau),'o', label = 'N = 100')
plt.plot(x3centers/200**d,y3*(x3centers**tau),'o', label = 'N = 200')
plt.hlines(0.5, 0,0.05, color = 'k', linestyles='dashed')
plt.vlines(0.05, 0, 0.5, color = 'k', linestyles= 'dashed')
#plt.plot(x0centers/(25**2.5), y0*(x0centers**1.2), 'o', label = 'N = 25')
#plt.plot(x1centers/(50**2.86), y1*(x1centers**1.3), 'o', label = 'N = 50')
#plt.plot(x2centers/(100**2.86), y2*(x2centers**1.3), 'o', label = 'N = 100')
#plt.plot(x3centers/(200**2.86), y3*(x3centers**1.3), 'o', label = 'N = 200')
plt.xscale('log')
plt.yscale('log')
plt.title('Powerlaws for diff. system sizes')
plt.xlabel('log(s/L^D)')
plt.ylabel('log(P(s)*s^tau)')
plt.xlim(0, 30)
plt.legend()
plt.show()
plt.close('All')





# %%
print(10e0)
# %%
