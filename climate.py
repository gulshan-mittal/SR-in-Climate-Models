# Program to numerically integrate the dynamics of a particle in a double-well potential and plot the trajectory and the power spectrum.

import numpy
from matplotlib import pyplot as plt
from matplotlib import mlab
import scipy.integrate
import scipy.signal
import math
import sys

dt = 0.1
T = 5000
sampling_rate = 1/dt
t = numpy.arange(0, T, dt)
N = len(t)
A = 0.3
v0 = 0.005
sigma = 0.8
noise = numpy.random.normal(0.0, sigma, N)

def climate(x, time, A, v0, n):
	#print(x, time, A, v0, n)
	x_dot = x - x**3 + A*math.sin(2*math.pi*v0*time) + n
	#print(x_dot)
	#input("")
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

X = numpy.zeros(N)
X[0] = 0.0
X = integrate_RK(X)

freqs, P_xx = scipy.signal.periodogram(X, sampling_rate, scaling='density')


# Plot trajectory
plt.rc('font', family='STIXGeneral', size=15)
plt.rc('xtick')
plt.rc('ytick')
plt.figure()
plt.plot(t, X, color='red', linewidth=1.5)
plt.plot(t, A*numpy.sin(2*math.pi*v0*t), color='green', linewidth=1.3)
plt.xlabel('Time')
plt.ylabel('X')
plt.xlim(0, T)
plt.figure()
plt.plot(freqs, P_xx, color='red', linewidth=1.0)
plt.xlim(0, 2*v0)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.show()