# Calculate B2U rectifier input current decomposition, THD and power factor and save it to a csv file
# (Required math operations resulted in mathtikz errors, so numpy was used instead)

import numpy as np
import os

# define operation point variables
beta = 0.7*np.pi
alpha = np.arctan((1-np.cos(beta))/(beta-np.sin(beta)))
gain = np.sin(alpha)

# define input current with x = omega*t considering the interval x=0...2*pi
def i_1(x):
    if 0 < x < alpha:
        return 0
    elif alpha < x < alpha+beta:
        return np.pi/2 * (np.cos(alpha)-np.cos(x)-gain*(x-alpha))
    elif alpha < x-np.pi < alpha+beta:
        return -np.pi/2 * (np.cos(alpha)-np.cos(x-np.pi)-gain*(x-alpha-np.pi))
    else:
        return 0
    
# define the interval x=0...2*pi with 360 points
xn = 360
x = np.linspace(0, 2*np.pi, xn)

# calculate the Fourier coefficients of i_1 for a specific order k
def fourier_coefficients(k):
    a = 1/np.pi * np.trapz([i_1(xi)*np.cos(k*xi) for xi in x],x)
    b = 1/np.pi * np.trapz([i_1(xi)*np.sin(k*xi) for xi in x],x)
    return a,b

# calculate the first 50 Fourier coefficients in a list
fn = 100
fourier = [fourier_coefficients(k) for k in range(0,fn-1)]

# Define the input current fundamental component
def i_1_fundamental(x):
    return fourier[1][0]*np.cos(x)+fourier[1][1]*np.sin(x)

# Define the harmonic input curent
def i_1_harmonic(x):
    return i_1(x) - i_1_fundamental(x)


# plot the input current and its fundamental component as well as the harmonic components
# import matplotlib.pyplot as plt
# plt.plot(x, [i_1(xi) for xi in x], label='i_1')
# plt.plot(x, [i_1_fundamental(xi) for xi in x], label='i_1 fundamental')
# plt.plot(x, [i_1_harmonic(xi) for xi in x], label='i_1 harmonic')
# plt.xlabel('x [rad]')
# plt.ylabel('i_1 [A]')
# plt.legend()
# plt.grid()
# plt.show()

# Save the current decomposition to a csv file
current_directory = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_directory, 'Grid_current_B2U_decomposition.csv')
np.savetxt(save_path, np.column_stack((x, [i_1(xi) for xi in x], [i_1_fundamental(xi) for xi in x], [i_1_harmonic(xi) for xi in x])), delimiter=',', header='wt, i_1, i_1_fundamental, i_1_harmonic', comments='')
