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
a=np.random.randint(5,size = (500,500))
for i in range(100):
	a100.append(a)
	a = Soc(a)





# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 500), ylim=(0, 500))

a=np.random.randint(5,size = (500,500))
im=plt.imshow(a,interpolation='none')
plt.colorbar(im)

# initialization function: plot the background of each frame
def init():
    im.set_data(np.random.randint(5,size = (500,500)))
    return [im]

# animation function.  This is called sequentially
def animate(i):
    a=im.get_array()
    #a=a*Soc(500,500,i)
    a = Soc(a)   
    im.set_array(a)
    return [im]

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

#anim.save('SandPile_animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'])

print('Done!')
#plt.show()


def Soc(L, N, T):
	lattice = np.zeros((N+2, N+2))
	lattice[1:N+1, 1:N+1] = np.random.randint(5,size = (L,L))

	for t in range(0,T):
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