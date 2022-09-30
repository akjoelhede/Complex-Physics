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
a=np.random.randint(4,size = (200,200))
for i in range(1000):
	a100.append(a)
	a = Soc(a)

img = plt.imshow(a100[1])
plt.colorbar(img)
plt.show()

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 200), ylim=(0, 200))


im=plt.imshow(a100[0],interpolation='none')
plt.colorbar(im)


# animation function.  This is called sequentially
def animate(i):
	new_data = a100[i]
	im.set_array(new_data)
	return [im]

anim = animation.FuncAnimation(fig, animate, frames=1000, interval=20, blit=True)

anim.save('SandPile_animation.mp4', fps=100, extra_args=['-vcodec', 'libx264'])

print('Done!')
plt.show()