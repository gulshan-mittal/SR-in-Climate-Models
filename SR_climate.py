# Program to plot the signal power (at driving frequency) as a function of noise intensity. Peak implies Stochastic Resonance
import numpy
from matplotlib import pyplot as plt
import scipy.signal
import math
import sys

dt = 0.1
T = 10000
t = numpy.arange(0, T, dt)
N = len(t)
A = 0.3
v0 = 0.005
sampling_rate = 1/dt


def climate(x, time, A, v0, n):
	x_dot = x - x**3 + A*math.sin(2*math.pi*v0*time) + n
	return x_dot

# Fourth order Runge-Kutta integrator
def integrate_RK(X):
	for i in range(N-1):
		k1 = dt*climate(X[i], t[i], A, v0, noise[i])
		k2 = dt*climate(X[i]+0.5*k1, t[i]+0.5*dt, A, v0, noise[i])
		k3 = dt*climate(X[i]+0.5*k2, t[i]+0.5*dt, A, v0, noise[i])
		k4 = dt*climate(X[i]+k3, t[i]+dt, A, v0, noise[i])
		X[i+1] = X[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
		#print(X[i+1], t[i+1]) 
		#input("")
	return X

intensity = numpy.arange(0.5, 2.0, 0.04)
sr_measure = numpy.zeros(len(intensity))

for j in range(len(intensity)):
	print('Iteration: '+str(j+1)+'/'+str(len(intensity)))	
	sigma = intensity[j]
	noise = numpy.random.normal(0.0, sigma, N)
	X = numpy.zeros(N)

	X = integrate_RK(X)
	
	freqs, P_xx = scipy.signal.periodogram(X, sampling_rate, scaling='density')
	f_nn, P_nn = scipy.signal.periodogram(noise, sampling_rate, scaling='density')
	d = freqs[2] - freqs[1]
	print(freqs[2] - freqs[1])
	print(freqs[numpy.where(freqs == v0)[0][0]])
	foo = numpy.where(numpy.logical_and(freqs>=v0-(15*d), freqs<=v0+(15*d)))
	sig_pow = numpy.trapz(P_xx[foo], dx = d)
	n_pow = numpy.trapz(P_nn[foo], dx = d)
	# Use the below for plotting signal power. Comment the line below it.
	#sr_measure[j] = 10*math.log10(P_xx[numpy.where(freqs == v0)[0][0]]/P_nn[numpy.where(freqs == v0)[0][0]])
	sr_measure[j] = 10*math.log10(sig_pow/n_pow)

plt.rc('font', family='STIXGeneral', size=15)
plt.rc('xtick')
plt.rc('ytick')
plt.figure()
plt.plot(intensity, sr_measure, marker='o', color='r', markersize=6, fillstyle='none', markeredgewidth=1.5, linestyle='none')
plt.xlabel('Noise intensity (standard deviation)')
plt.ylabel('Signal to Noise ratio')
plt.show()