import os

import numpy as np
from scipy.signal import find_peaks

###################################################
# Some parameters #
###################################################

# define the interval x=0...2*pi with xn points
xn = 10000
x = np.linspace(0, 2 * np.pi, xn)

# pulse number (number of carrier periods per fundamental period)
N = 10

###################################################
# Helper functions / function definitions #
###################################################


# generate a sawtooth carrier sequence c(w*t) between +1, -1
def c_saw(wt, N):
    return 1 - 2 * np.abs(np.modf((wt * N) / (2 * np.pi))[0])


# generate a triangular carrier sequence c(w*t) between +1, -1
def c_tri(wt, N):
    return 4 * (np.abs(np.modf((wt * N) / (2 * np.pi))[0] - 0.5)) - 1


# generate a fundamental and normalized sinusoidal reference d(t)
def d_sin(wt):
    return np.sin(wt)


# generate a fundamental and normalized sinusoidal reference d(t) with overmod. amplitude
def d_sin_overmod(wt):
    return 1.18 * np.sin(wt)


# calculate complementary PWM-based switching signals
def s_comp(d, c):
    return [np.where(d > c, 1, -1), np.where(d > c, -1, 1)]


# calculate interleaved PWM-based switching signals
def s_int(d, c):
    return [np.where(d > c, 1, -1), np.where(-d > c, 1, -1)]


# calculate the integrated / summed error between the reference and the switching signal
def e(d, s, xn):
    return np.cumsum(d - s) / xn * 2 * np.pi


###################################################
# Complementary switching PWM example #
###################################################

c_comp_example = c_tri(x, N)
d_comp_example = d_sin(x)
s_comp_example = s_comp(d_comp_example, c_comp_example)
e_comp_example = e(d_comp_example, (s_comp_example[0] - s_comp_example[1]) / 2, xn)

# Compute the derivative of the signal
s_diff = np.diff((s_comp_example[0] - s_comp_example[1]) / 2)

# Identify step indices (where derivative is non-zero)
step_indices = np.where(s_diff != 0)[0]

# Find the two nearest samples for each step
nearest_samples = [0]
for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step + 1)

# identify peaks of the carrier
upper_peaks, _ = find_peaks(c_comp_example)
lower_peaks, _ = find_peaks(-c_comp_example)

# combine the three sets of indices and add the first and last sample
idx_sum = np.unique(
    np.concatenate((nearest_samples, upper_peaks, lower_peaks, [0, len(x) - 1]))
)

# add n additional samples within idx_sum such that the resulting distances between samples is as equal as possible
# to do this find the biggest distance between two samples and add a sample index in the middle of the two samples
# repeat this until n samples are added using a while loop
n = 100
while n > 0:
    # find the biggest distance between two samples
    max_dist = np.max(np.diff(idx_sum))
    # find the indices of the biggest distance
    max_dist_idx = np.argmax(np.diff(idx_sum))
    # add a sample index in the middle of the two samples
    idx_sum = np.insert(
        idx_sum,
        max_dist_idx + 1,
        (idx_sum[max_dist_idx] + idx_sum[max_dist_idx + 1]) // 2,
    )
    # decrement n
    n -= 1

# save the reduced data to a csv file
current_directory = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_directory, "PWM_single_phase_comp_example.csv")
np.savetxt(
    save_path,
    np.column_stack(
        (
            x[idx_sum],
            s_comp_example[0][idx_sum],
            s_comp_example[1][idx_sum],
            (s_comp_example[0][idx_sum] - s_comp_example[1][idx_sum]) / 2,
            d_comp_example[idx_sum],
            c_comp_example[idx_sum],
            e_comp_example[idx_sum],
        )
    ),
    delimiter=",",
    header="wt, s1, s2, s, d, c, e",
    comments="",
)

###################################################
# Interleaved switching PWM example #
###################################################

c_int_example = c_tri(x, N)
d_int_example = d_sin(x)
s_overmod_example = s_int(d_int_example, c_int_example)
e_int_example = e(d_int_example, (s_overmod_example[0] - s_overmod_example[1]) / 2, xn)

# Compute the derivative of the signal
s_diff = np.diff((s_overmod_example[0] - s_overmod_example[1]) / 2)

# Identify step indices (where derivative is non-zero)
step_indices = np.where(s_diff != 0)[0]

# Find the two nearest samples for each step
nearest_samples = [0]
for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step + 1)

# identify peaks of the carrier
upper_peaks, _ = find_peaks(c_int_example)
lower_peaks, _ = find_peaks(-c_int_example)

# combine the three sets of indices and add the first and last sample
idx_sum = np.unique(
    np.concatenate((nearest_samples, upper_peaks, lower_peaks, [0, len(x) - 1]))
)

# add n additional samples within idx_sum such that the resulting distances between samples is as equal as possible
# to do this find the biggest distance between two samples and add a sample index in the middle of the two samples
# repeat this until n samples are added using a while loop
n = 100

while n > 0:
    # find the biggest distance between two samples
    max_dist = np.max(np.diff(idx_sum))
    # find the indices of the biggest distance
    max_dist_idx = np.argmax(np.diff(idx_sum))
    # add a sample index in the middle of the two samples
    idx_sum = np.insert(
        idx_sum,
        max_dist_idx + 1,
        (idx_sum[max_dist_idx] + idx_sum[max_dist_idx + 1]) // 2,
    )
    # decrement n
    n -= 1

# save the reduced data to a csv file
current_directory = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_directory, "PWM_single_phase_int_example.csv")
np.savetxt(
    save_path,
    np.column_stack(
        (
            x[idx_sum],
            s_overmod_example[0][idx_sum],
            s_overmod_example[1][idx_sum],
            (s_overmod_example[0][idx_sum] - s_overmod_example[1][idx_sum]) / 2,
            d_int_example[idx_sum],
            c_int_example[idx_sum],
            e_int_example[idx_sum],
        )
    ),
    delimiter=",",
    header="wt, s1, s2, s, d, c, e",
    comments="",
)

###################################################
# Interleaved switching PWM example with overmod.#
###################################################

c_overmod_example = c_tri(x, N)
d_overmod_example = d_sin_overmod(x)
s_overmod_example = s_int(d_overmod_example, c_overmod_example)
e_overmod_example = e(
    d_overmod_example, (s_overmod_example[0] - s_overmod_example[1]) / 2, xn
)

# Compute the derivative of the signal
s_diff = np.diff((s_overmod_example[0] - s_overmod_example[1]) / 2)

# Identify step indices (where derivative is non-zero)
step_indices = np.where(s_diff != 0)[0]

# Find the two nearest samples for each step
nearest_samples = [0]
for step in step_indices:
    nearest_samples.append(step)
    nearest_samples.append(step + 1)

# identify peaks of the carrier
upper_peaks, _ = find_peaks(c_overmod_example)
lower_peaks, _ = find_peaks(-c_overmod_example)

# combine the three sets of indices and add the first and last sample
idx_sum = np.unique(
    np.concatenate((nearest_samples, upper_peaks, lower_peaks, [0, len(x) - 1]))
)

# add n additional samples within idx_sum such that the resulting distances between samples is as equal as possible
# to do this find the biggest distance between two samples and add a sample index in the middle of the two samples
# repeat this until n samples are added using a while loop
n = 100

while n > 0:
    # find the biggest distance between two samples
    max_dist = np.max(np.diff(idx_sum))
    # find the indices of the biggest distance
    max_dist_idx = np.argmax(np.diff(idx_sum))
    # add a sample index in the middle of the two samples
    idx_sum = np.insert(
        idx_sum,
        max_dist_idx + 1,
        (idx_sum[max_dist_idx] + idx_sum[max_dist_idx + 1]) // 2,
    )
    # decrement n
    n -= 1

# save the reduced data to a csv file
current_directory = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_directory, "PWM_single_phase_overmod_example.csv")
np.savetxt(
    save_path,
    np.column_stack(
        (
            x[idx_sum],
            s_overmod_example[0][idx_sum],
            s_overmod_example[1][idx_sum],
            (s_overmod_example[0][idx_sum] - s_overmod_example[1][idx_sum]) / 2,
            d_overmod_example[idx_sum],
            c_overmod_example[idx_sum],
            e_overmod_example[idx_sum],
        )
    ),
    delimiter=",",
    header="wt, s1, s2, s, d, c, e",
    comments="",
)

###################################################

# generate an example carrier signal and plot it for N=10
import matplotlib.pyplot as plt

plt.plot(x, c_tri(x, N))
plt.plot(x, d_sin_overmod(x))
plt.xlabel(r"$\omega t$")
plt.ylabel(r"$c(\omega t)$")
plt.title("Triangular carrier signal")
plt.grid()
plt.show()

# add subplot for switching signal
plt.plot(
    x,
    (s_int(d_sin_overmod(x), c_tri(x, N))[0] - s_int(d_sin_overmod(x), c_tri(x, N))[1])
    / 2,
)
plt.xlabel(r"$\omega t$")
plt.ylabel(r"$s(\omega t)$")
plt.title("Switching signal")
plt.grid()
plt.show()

# add subplot for error signal
plt.plot(
    x,
    e(
        d_sin_overmod(x),
        (
            s_int(d_sin_overmod(x), c_tri(x, N))[0]
            - s_int(d_sin_overmod(x), c_tri(x, N))[1]
        )
        / 2,
        xn,
    ),
)
plt.xlabel(r"$\omega t$")
plt.ylabel(r"$e(\omega t)$")
plt.title("Error signal")
plt.grid()
plt.show()
