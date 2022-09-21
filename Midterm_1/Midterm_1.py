import numpy as np
import matplotlib.pyplot as plt
from  tqdm import tqdm
from numba import njit



"First implementation of 1D spin chain where each spin site has 2 neighbors, on to the left and one to the right and"

@njit
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

@njit
def calc_energy(state):
    nbor_magnetization = np.roll(state, 1) + np.roll(state, -1) + np.roll(state, -2) + np.roll(state, 2)
    return -np.sum(state * nbor_magnetization) / 2

@njit
def calc_energy4N(state):
	Nearest_con = np.roll(state, 1) + np.roll(state, -1)
	M_con = np.zeros_like(state)
	for i in range(len(state)):
		M_con[i] += state[neigh_4N[i][0]] + state[neigh_4N[i][1]]

	E = -0.5*np.sum(Nearest_con*M_con*state)
	return E

@njit
def update(state, beta):
    for _ in range(state.size):
        rand_x = np.random.randint(state.shape[0]) 
        new_state = state.copy()
        new_state[rand_x] *= -1
        dE = calc_energy(new_state) - calc_energy(state)
        
        if np.exp(-dE*beta) > np.random.rand():
            state[:] = new_state # only with the probability we keep the new state

@njit
def calc_stat(state):
    return np.abs(state.mean()), calc_energy(state)

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


all_states_energy = run_multiple1(1/betas, shape, n_warmup=0, n_average=100)
plt.plot(all_states_energy[:,:,1].T, alpha = 0.5)
plt.savefig("1D_Ising_a_multipletemp_3.pdf")
plt.close('all')

all_states_avgenergy = run_multiple1(1/betas, shape, n_warmup=0, n_average=100)
plt.violinplot(all_states_avgenergy[:,:,1].T, betas, widths=0.02, showextrema=False, showmeans=True);
plt.title("Average Energy vs Temperature")
plt.ylabel(r"Average energy per spin ")
plt.xlabel(r"Temperature ($k_b T/J$)")
plt.savefig('Avg_Energy_a_3.pdf')
plt.close('all')

all_states_avgmag = run_multiple1(1/betas, shape, n_warmup=10, n_average=100)
plt.violinplot(all_states_avgmag[:,:,0].T, betas, widths=0.02, showextrema=False, showmeans=True);
plt.title("Average Magnetisation vs Temperature")
plt.ylabel(r"Average Magnetisation per spin ")
plt.xlabel(r"Temperature ($k_b T/J$)")
plt.savefig('Avg_Magnetisation_a_3.pdf')
plt.close('all')

sys_N = [1,50,200,1000]

fig, ax  = plt.subplots(nrows = len(sys_N), sharex=True)
for n in tqdm(range(len(sys_N))):
	state_mag = run_multiple1(1/betas, sys_N[n], n_warmup=10, n_average=100)
	ax[n].violinplot(np.abs(state_mag[:,:,1].T), betas, widths=0.02, showextrema=False, showmeans=True)
	ax[n].set_title(f'For N = {sys_N[n]}')
	if n == (len(sys_N)-1):
		ax[n].set_xlabel('T')
		ax[n].set_ylabel('Absolute Average Magnetisation|m|')
fig.tight_layout()
fig.savefig("diff_N_a_3.pdf")
plt.close('all')