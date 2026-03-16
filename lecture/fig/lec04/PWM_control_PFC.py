import os

import numpy as np
from scipy.signal import find_peaks

###################################################
# Some parameters #
###################################################

# define angular switching frequency as a fraction of 2*pi (grid period)
delta_omegaTs = 2 * np.pi / 50

# define min. output to input voltage ratio M
M = 1.3

# further parameters
L = 2e-1
omega = 50 * 2 * np.pi
iamp = 3
uamp = 10

# define the interval x=0...pi with xn points
xn = 10000
x = np.linspace(0.001, np.pi - 0.001, xn)

###################################################
# Helper functions / function definitions #
###################################################


# generate a triangular switching carrier sequence c(t) with period delta_omegaTs
def c_tri(x, delta_omegaTs):
    return 1 - 2 * np.abs(np.modf(x / delta_omegaTs)[0] - 0.5)


# generate a sawtooth switching carrier sequence c(t) with period delta_omegaTs and a leading edge at x=0
def c_saw(x, delta_omegaTs):
    return np.modf(x / delta_omegaTs)[0]


# generate a sawtooth switching carrier sequence c(t) with period delta_omegaTs and a falling edge at x=0
def c_saw_falling(x, delta_omegaTs):
    return 1 - np.modf(x / delta_omegaTs)[0]


# Reference signal d(t) with min voltage transfer ratio M for boost-based PFC
def d_PFC(x, M):
    return 1 + L * iamp / M / uamp * np.cos(x) - np.sin(x) / M


# Reference signal d(t) linearily increasing from 0 to 1 within x=0...pi
def d_lin(x):
    return x / np.pi


# calculate a PWM-based switching signal comparing d(t)-c(t) within a comparator
def s(d, c):
    return np.where(d > c, 1, 0)


# calculate the current response assuming an ideal integrator behavior with initial state x0 based on the switching signal s(t). Also i_1 cannot get negative due to boost converter topology.
def i_1(s, x0, M, x):
    i = np.zeros(len(s))
    i[0] = x0
    for n in range(1, len(s)):
        i[n] = i[n - 1] + uamp * (np.sin(x[n]) - (1 - s[n]) * M) / L * np.pi / xn
        if i[n] < 0:
            i[n] = 0
    return i


###################################################
# Open-loop PFC-boost PWM example #
###################################################

d_PFC_sample = d_PFC(x, M)
c_PFC_sample = c_tri(x, delta_omegaTs)
s_PFC_sample = s(d_PFC_sample, c_PFC_sample)
i1_PFC_sample = i_1(s_PFC_sample, 0.001, M, x)

# normalize the current w.r.t to its maximum value
i1_ref = iamp * np.sin(x) / np.max(i1_PFC_sample)
i1_PFC_sample = i1_PFC_sample / np.max(i1_PFC_sample)
i1_delta = i1_PFC_sample - i1_ref
i1_delta = i1_delta - np.mean(i1_delta)

# Compute the derivative of the signal
s_diff = np.diff(s_PFC_sample)

# Identify step indices (where derivative is non-zero)
step_indices = np.where(s_diff != 0)[0]

# Find the two nearest samples for each step
nearest_samples = [0]
for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step + 1)


# identify peaks of the carrier
upper_peaks, _ = find_peaks(c_PFC_sample)
lower_peaks, _ = find_peaks(-c_PFC_sample)

# combine the three sets of indices and add the first and last sample
idx_sum = np.unique(
    np.concatenate((nearest_samples, upper_peaks, lower_peaks, [0, len(x) - 1]))
)

# save the reduced data to a csv file
current_directory = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_directory, "PWM_PFC_example.csv")
np.savetxt(
    save_path,
    np.column_stack(
        (
            x[idx_sum],
            s_PFC_sample[idx_sum],
            d_PFC_sample[idx_sum],
            c_PFC_sample[idx_sum],
            i1_PFC_sample[idx_sum],
            i1_ref[idx_sum],
            i1_delta[idx_sum],
        )
    ),
    delimiter=",",
    header="wt, s, d, c, i1, i1ref, i1delta",
    comments="",
)


###################################################
# Dummy PWM with linearly increasing duty cycle and triangular carrier #
###################################################

s_tri = s(d_lin(x), c_tri(x, delta_omegaTs))
c_tri_sample = c_tri(x, delta_omegaTs)
d_lin_sample = d_lin(x)

# Compute the derivative of the signal
s_diff = np.diff(s_tri)

# Identify step indices (where derivative is non-zero)
step_indices = np.where(s_diff != 0)[0]

# Find the two nearest samples for each step
nearest_samples = [0]
for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step + 1)

# identify peaks of the carrier
upper_peaks, _ = find_peaks(c_tri_sample)
lower_peaks, _ = find_peaks(-c_tri_sample)

# combine the three sets of indices and add the first and last sample
idx_sum = np.unique(
    np.concatenate((nearest_samples, upper_peaks, lower_peaks, [0, len(x) - 1]))
)

# save the reduced data to a csv file
current_directory = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_directory, "PWM_triangular_example.csv")
np.savetxt(
    save_path,
    np.column_stack(
        (x[idx_sum], s_tri[idx_sum], d_lin_sample[idx_sum], c_tri_sample[idx_sum])
    ),
    delimiter=",",
    header="wt, s, d, c",
    comments="",
)


###################################################
# Dummy PWM with linearly increasing duty cycle and sawtooth carrier #
###################################################
s_saw = s(d_lin(x), c_saw(x, delta_omegaTs))
c_saw_sample = c_saw(x, delta_omegaTs)

s_diff = np.diff(s_saw)
step_indices = np.where(s_diff < -0.5)[0]
nearest_samples = [0]

for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step + 1)

c_diff = np.diff(c_saw_sample)
step_indices = np.where(c_diff < -0.5)[0]
for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step + 1)

# sort nearest_samples in ascending order
nearest_samples = np.sort(np.concatenate((nearest_samples, [0, len(x) - 1])))

# combine the three sets of indices
idx_sum = np.unique(nearest_samples)

# save the reduced data to a csv file
current_directory = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_directory, "PWM_sawtooth_example.csv")
np.savetxt(
    save_path,
    np.column_stack(
        (x[idx_sum], s_saw[idx_sum], d_lin_sample[idx_sum], c_saw_sample[idx_sum])
    ),
    delimiter=",",
    header="wt, s, d, c",
    comments="",
)
