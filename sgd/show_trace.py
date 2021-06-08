import matplotlib.pyplot as plt
import numpy as np
def train_2d(trainer, y):
	x1, x2, s1, s2 = -5, -2, 0, 0
	results = [(x1, x2)]
	for i in range(20):
		x1, x2, s1, s2 = trainer(x1, x2, s1, s1)
		print('------' + str(i) + '---------')
		print(x1)
		print(x2)
		results.append((x1,x2))
	return results
def show_trace(f, results, y):
	plt.plot(*zip(*results), '-o', color = '#ff7f0e')
	x1,  x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
	plt.contour(x1, x2, f(x1, x2, y))
	plt.show()

def f(x1, x2):
	return 0.1 * x1 ** 2 + 2 * x2 **2

def f_2d(x1, x2, y):
	return (y - f(x1, x2))**2

def gd_2d(x1, x2, y, s1, s2):
	return (x1 - eta * 2 * (y - f(x1, x2)) * (2 * x1 * 0.1), x2-eta*2*(y - f(x1, x2)) * (2 * x2 * 2), 0,0)
eta = 0.4
show_trace(f_2d, train_2d(gd_2d))
