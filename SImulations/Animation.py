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
a=np.random.randint(6,size = (10,10))
for i in tqdm(range(1000)):
	a100.append(a)
	a = Soc(a)

print(a100[1].sum(), a100[999].sum())

fig = plt.figure(figsize = (8,8))
im = plt.imshow(a100[0],cmap = 'inferno')
plt.colorbar()
def animate_func(i):
	im.set_array(a100[i])
	return [im]

anim = animation.FuncAnimation(fig, animate_func, frames = 1000, interval = 20)
plt.show()
plt.close('All')
#img = plt.imshow(a100[999])
#plt.colorbar(img)
#plt.show()

def CenterSOC(L):#, N, T):
	N = len(L)
	lattice = np.zeros((N+2, N+2))
	#lattice[1:N+1, 1:N+1] = np.random.randint(5,size = (L,L))
	lattice[1:N+1, 1:N+1] = L

	Half = np.floor(N/2) + 1
	#for t in range(0,T):
	if np.all(lattice < 4 ):
		lattice[int(Half), int(Half)] += 1

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

b100 = []
b=np.zeros((45,45))
for i in tqdm(range(1000)):
	b100.append(b)
	b = CenterSOC(b)

print(b100[1].sum(), b100[999].sum())

fig_b = plt.figure(figsize = (8,8))
## vi laver en random "baggrund" for hvis vi bruger nullerne som første
## image så forstår colorbaren ikke hvad rangen er
c = np.random.randint(5, size = (45,45))
#im = plt.imshow(b100[0])
im = plt.imshow(c, cmap = 'inferno')
def animate_func_b(i):
	im.set_array(b100[i])
	return [im]

anim_b = animation.FuncAnimation(fig_b, animate_func_b, frames = 1000, interval = 100)
plt.colorbar()

anim_b.save('SandPile_CENTER_animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'])

plt.show()
#img = plt.imshow(a100[999])
#plt.colorbar(img)
#plt.show()