import numpy as np
import os
from scipy.signal import find_peaks

###################################################
# Some parameters #
###################################################

# define the interval x=0...2*pi with xn points
xn = 10000
x = np.linspace(0, 2*np.pi, xn)

# pulse number (number of carrier periods per fundamental period)
N = 10

###################################################
# Helper functions / function definitions #
###################################################

# generate a sawtooth carrier sequence c(w*t) between +1, -1
def c_saw(wt, N):
    return 1 - 2 * np.abs(np.modf((wt *N)/(2*np.pi))[0])

# generate a triangular carrier sequence c(w*t) between +1, -1
def c_tri(wt, N):
    return  4*(np.abs(np.modf((wt *N)/(2*np.pi))[0] - 0.5)) -1

# generate a fundamental and normalized sinusoidal reference d(t)
def d_sin(wt, phi = 0):
    return np.sin(wt - phi)

# generate a fundamental and normalized sinusoidal reference d(t) with overmod. amplitude
def d_sin_overmod(wt, phi = 0):
    return 1.18*np.sin(wt - phi)

# calculate complementary PWM-based switching signals 
def s_comp(d, c):
    return np.where(d>c, 1, -1)

#calculate the integrated / summed error between the reference and the switching signal
def e(d, s, xn):
    return np.cumsum(d - s)/xn*2*np.pi


###################################################
# Complementary switching PWM example #
###################################################

c_example = c_tri(x, N)
d_a_example = d_sin(x)
d_b_example = d_sin(x, np.pi/3*2)
d_c_example = d_sin(x, np.pi/3*4)
s_a_example = s_comp(d_a_example, c_example)
s_b_example = s_comp(d_b_example, c_example)
s_c_example = s_comp(d_c_example, c_example)

# e_comp_example = e(d_comp_example, (s_comp_example[0] - s_comp_example[1])/2, xn)

# Compute the derivative of the signal
s_diff = np.diff(s_a_example)

# Identify step indices (where derivative is non-zero)
step_indices = np.where(s_diff != 0)[0]

# Find the two nearest samples for each step
nearest_samples = [0]
for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step+1)

s_diff = np.diff(s_b_example)
step_indices = np.where(s_diff != 0)[0]
for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step+1)

s_diff = np.diff(s_c_example)
step_indices = np.where(s_diff != 0)[0]
for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step+1)

# identify peaks of the carrier
upper_peaks, _ = find_peaks(c_example)
lower_peaks, _ = find_peaks(-c_example)

#combine the three sets of indices and add the first and last sample
idx_sum = np.unique(np.concatenate((nearest_samples, upper_peaks, lower_peaks, [0, len(x)-1])))


# save the reduced data to a csv file
current_directory = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_directory, 'PWM_three-phase_mod1_example.csv')
np.savetxt(save_path, np.column_stack((x[idx_sum], s_a_example[idx_sum], s_b_example[idx_sum], s_c_example[idx_sum], d_a_example[idx_sum], d_b_example[idx_sum], d_c_example[idx_sum], c_example[idx_sum])), delimiter=',', header='wt, s1, s2, s3, d1, d2, d3, c', comments='')


###################################################

# generate an exmaple carrier signal and plot it for N=10
import matplotlib.pyplot as plt
plt.plot(x, c_example)
plt.plot(x, d_a_example)
plt.plot(x, d_b_example)
plt.plot(x, d_c_example)
plt.xlabel(r'$\omega t$')
plt.ylabel(r'$c(\omega t)$')
plt.title('Triangular carrier signal')
plt.grid()
plt.show()

# add subplot for switching signal
plt.plot(x, s_a_example)
plt.plot(x, s_c_example)
plt.plot(x, s_b_example)
plt.xlabel(r'$\omega t$')
plt.ylabel(r'$s(\omega t)$')
plt.title('Switching signal')
plt.grid()
plt.show()




