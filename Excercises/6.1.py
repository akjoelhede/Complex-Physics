from random import random
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit


x = []

for i in range(100000):
	x_store = random.random()
	x.append(x_store)

print(len(x))

def number_set(x):
	y = -np.log(x)
	return y

def P1(x,a,b):
	y = b * a**x
	return y


y_hist, x_hist = np.histogram(number_set(x), bins = 100)

xcenters = (x_hist[:-1] + x_hist[1:])/2

par, cov = curve_fit(P1, xcenters, y_hist)
print("Det her er parametrene, og cov matricens diagonal", par, np.sqrt(np.diag(cov)))


plt.plot(xcenters, y_hist, label = 'x_i')
plt.plot(xcenters, P1(xcenters, *par), label = 'Exponential fit')
plt.legend()
plt.show()

