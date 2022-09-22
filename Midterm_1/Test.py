import numpy as np
import matplotlib.pyplot as plt
from  tqdm import tqdm
from numba import njit



"First implementation of 1D spin chain where each spin site has 2 neighbors, on to the left and one to the right and"


def ran_neigh(M):
	Random_neighbours = []
	for i in range(len(M)):


		if i == 0:
			r_n = i + 1
			l_n = len(M)-1
		
		if i == len(M)-1:
			r_n = 0
			l_n = i-1

		else:
			r_n = i+1
			l_n = i-1


		rand_neig1 = np.random.choice(len(M))
		while rand_neig1 == i or rand_neig1 == l_n or rand_neig1 == r_n:
			rand_neig1=np.random.choice(len(M))

		rand_neig2 = np.random.choice(len(M))
		while rand_neig2 == i or rand_neig2 == l_n or rand_neig2 == r_n or rand_neig2 == rand_neig1:
			rand_neig2 = np.random.choice(len(M))

		Random_neighbours.append(np.array([rand_neig1, rand_neig2]))


	return Random_neighbours



default_settings = [np.linspace(2,2.5,20), (200), 0]

def init(shape):
    return np.random.choice([-1,1], shape)

state_0 = init(default_settings[1])
neigh_4N = ran_neigh(state_0)


def calc_energy4N(state):
	Nearest_con = np.roll(state, 1) + np.roll(state, -1)
	M_con = np.zeros_like(state)
	for i in range(len(state)):
		M_con[i] += state[neigh_4N[i][0]] + state[neigh_4N[i][1]]

	E = -0.5*np.sum(Nearest_con*M_con*state)
	return E


def update(state, beta):
    for _ in range(state.size):
        rand_x = np.random.randint(state.shape[0]) 
        new_state = state.copy()
        new_state[rand_x] *= -1
        dE = calc_energy4N(new_state) - calc_energy4N(state)
        
        if np.exp(-dE*beta) > np.random.rand():
            state[:] = new_state # only with the probability we keep the new state


def calc_stat(state):
    return np.abs(state.mean()), calc_energy4N(state)

def run_simulation(shape, beta, n_warmup, n_average):
    state = init(shape)
    stats = []
    for i in range(n_warmup + n_average):
        update(state, beta)
        if i >= n_warmup:
            stats.append(calc_stat(state))
    return stats


def run_multiple1(betas, shape, n_warmup, n_average):
    all_states = []
    for beta in tqdm(betas):
        all_states.append(run_simulation(shape, beta, n_warmup, n_average))
    return np.array(all_states)

beta = 2.3
betas = np.linspace(0.1,4, 40)
shape = 200

print(calc_energy4N(state_0))
plt.imshow(np.expand_dims(state_0, axis=0), aspect = 'auto')
plt.savefig("initial_state.pdf")
plt.close('all')

all_states_energy = run_multiple1(1/betas, shape, n_warmup=0, n_average=100)
plt.plot(all_states_energy[:,:,1].T, alpha = 0.5)
plt.savefig("1D_Ising_b_multipletemp.pdf")
plt.close('all')

all_states_avgenergy = run_multiple1(1/betas, shape, n_warmup=0, n_average=100)
plt.violinplot(all_states_avgenergy[:,:,1].T, betas, widths=0.02, showextrema=False, showmeans=True);
plt.title("Average Energy vs Temperature")
plt.ylabel(r"Average energy per spin ")
plt.xlabel(r"Temperature ($k_b T/J$)")
plt.savefig('Avg_Energy_b.pdf')
plt.close('all')

all_states_avgmag = run_multiple1(1/betas, shape, n_warmup=10, n_average=100)
plt.violinplot(all_states_avgmag[:,:,0].T, betas, widths=0.02, showextrema=False, showmeans=True);
plt.title("Average Magnetisation vs Temperature")
plt.ylabel(r"Average Magnetisation per spin ")
plt.xlabel(r"Temperature ($k_b T/J$)")
plt.savefig('Avg_Magnetisation_b.pdf')
plt.close('all')

sys_N = [1,50,200,1000]

def absmag(N):
	state_mag = run_multiple1(1/betas, N, n_warmup=10, n_average=100)
	return np.abs(state_mag[:,:,0].T)

fig, ax = plt.subplots(1)
ax.violinplot(absmag(200), betas, widths=0.02, showextrema=False, showmeans=True)
ax.set_title(f'For N = {200}')
ax.set_xlabel('T')
fig.tight_layout()
fig.savefig("diff_N_b.pdf")
plt.close('all')

