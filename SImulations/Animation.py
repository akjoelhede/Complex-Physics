import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


#input: L = initial lattice
def Soc(L):#, N, T):
	N = len(L)
	lattice = np.zeros((N+2, N+2))
	#lattice[1:N+1, 1:N+1] = np.random.randint(5,size = (L,L))
	lattice[1:N+1, 1:N+1] = L

	#for t in range(0,T):
	if np.all(lattice < 4 ):
		r1 = np.random.randint(1,N)
		r2 = np.random.randint(1,N)
		lattice[r1,r2]+=1

	else:
		add = np.zeros((N+2, N+2))

		for i in range(1, len(lattice)-1):
			for j in range(1, len(lattice)-1):
				if lattice[i,j]>=4:
					add[i,j] -= 4
					add[i+1,j] += 1
					add[i,j+1] += 1
					add[i-1,j] += 1
					add[i,j-1] += 1
		lattice = lattice + add
	return lattice[1:N+1, 1:N+1]

a100 = []
a=np.random.randint(6,size = (100,100))
for i in tqdm(range(1000)):
	a100.append(a)
	a = Soc(a)

print(a100[1].sum(), a100[999].sum())

fig = plt.figure(figsize = (8,8))
im = plt.imshow(a100[0])
plt.colorbar()
def animate_func(i):
	im.set_array(a100[i])
	return [im]

anim = animation.FuncAnimation(fig, animate_func, frames = 1000, interval = 20)
plt.show()
#img = plt.imshow(a100[999])
#plt.colorbar(img)
#plt.show()

