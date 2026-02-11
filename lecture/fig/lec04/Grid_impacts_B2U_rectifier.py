# Calculate B2U rectifier input current decomposition, THD and power factor and save it to a csv file
# (Required math operations resulted in mathtikz errors, so numpy was used instead)

import os

import numpy as np


# define input current with x = omega*t considering the interval x=0...2*pi (DCM)
def i_1(x, alpha, beta, gain):
    if alpha + beta < np.pi:
        if 0 < x < alpha:
            return 0
        elif alpha < x < alpha + beta:
            return np.pi / 2 * (np.cos(alpha) - np.cos(x) - gain * (x - alpha))
        elif alpha < x - np.pi < alpha + beta:
            return (
                -np.pi
                / 2
                * (np.cos(alpha) - np.cos(x - np.pi) - gain * (x - alpha - np.pi))
            )
        else:
            return 0
    else:
        if 0 <= x < beta + alpha - np.pi:
            return (
                -np.pi
                / 2
                * (np.cos(alpha) - np.cos(x + np.pi) - gain * (x - alpha + np.pi))
            )
        elif beta + alpha - np.pi < x < alpha:
            return 0
        elif alpha < x < alpha + beta:
            return np.pi / 2 * (np.cos(alpha) - np.cos(x) - gain * (x - alpha))
        elif alpha < x - np.pi <= alpha + beta:
            return (
                -np.pi
                / 2
                * (np.cos(alpha) - np.cos(x - np.pi) - gain * (x - alpha - np.pi))
            )
        else:
            return 0


# define input current with x = omega*t considering the interval x=0...2*pi (BCM)
def i_1_BCM(x, alpha_dash, gain_dash):
    if 0 <= x < alpha_dash:
        return (
            -np.pi
            / 2
            * (
                np.cos(alpha_dash)
                - np.cos(x + np.pi)
                - gain_dash * (x - alpha_dash + np.pi)
            )
        )
    elif alpha_dash < x < alpha_dash + np.pi:
        return (
            np.pi / 2 * (np.cos(alpha_dash) - np.cos(x) - gain_dash * (x - alpha_dash))
        )
    elif alpha_dash + np.pi < x <= 2 * np.pi:
        return (
            -np.pi
            / 2
            * (
                np.cos(alpha_dash)
                - np.cos(x - np.pi)
                - gain_dash * (x - alpha_dash - np.pi)
            )
        )
    else:
        return 0


# define the interval x=0...2*pi with 360 points
xn = 360
x = np.linspace(0, 2 * np.pi, xn)


# calculate the Fourier coefficients of i_1 for a specific order k (DCM)
def fourier_coefficients(k, alpha, beta, gain):
    a = (
        1
        / np.pi
        * np.trapz([i_1(xi, alpha, beta, gain) * np.cos(k * xi) for xi in x], x)
    )
    b = (
        1
        / np.pi
        * np.trapz([i_1(xi, alpha, beta, gain) * np.sin(k * xi) for xi in x], x)
    )
    return a, b


# calculate the Fourier coefficients of i_1 for a specific order k (BCM)
def fourier_coefficients_BCM(k, alpha_dash, gain_dash):
    a = (
        1
        / np.pi
        * np.trapz([i_1_BCM(xi, alpha_dash, gain_dash) * np.cos(k * xi) for xi in x], x)
    )
    b = (
        1
        / np.pi
        * np.trapz([i_1_BCM(xi, alpha_dash, gain_dash) * np.sin(k * xi) for xi in x], x)
    )
    return a, b


# Define the input current fundamental component (DCM)
def i_1_fundamental(x, alpha, beta, gain):
    return fourier_coefficients(1, alpha, beta, gain)[0] * np.cos(
        x
    ) + fourier_coefficients(1, alpha, beta, gain)[1] * np.sin(x)


# Define the input current fundamental component (BCM)
def i_1_BCM_fundamental(x, alpha_dash, gain_dash):
    return fourier_coefficients_BCM(1, alpha_dash, gain_dash)[0] * np.cos(
        x
    ) + fourier_coefficients_BCM(1, alpha_dash, gain_dash)[1] * np.sin(x)


# Define the harmonic input current component (DCM)
def i_1_harmonic(x, alpha, beta, gain):
    return i_1(x, alpha, beta, gain) - i_1_fundamental(x, alpha, beta, gain)


# Define the harmonic input current component (DCM)
def i_1_rms(x, alpha, beta, gain):
    return np.sqrt(
        1 / (2 * np.pi) * np.trapz([i_1(xi, alpha, beta, gain) ** 2 for xi in x], x)
    )


# Define the harmonic input current component (BCM & DCM)
def i_1_THD(fourier):
    return np.sqrt(
        np.sum([fourier[k][0] ** 2 + fourier[k][1] ** 2 for k in range(2, fn - 1)])
        / (fourier[1][0] ** 2 + fourier[1][1] ** 2)
    )


# Define the phase shift between the fundamental current and voltage component (BCM & DCM)
def phi(fourier):
    return (
        np.arccos(fourier[1][0] / np.sqrt(fourier[1][1] ** 2 + fourier[1][0] ** 2))
        * np.sign(fourier[1][1])
        - np.pi / 2
    )


# Define the power factor (BCM & DCM)
def lamb(fourier):
    return np.cos(phi(fourier)) / (np.sqrt(1 + i_1_THD(fourier) ** 2))


# Set operating point variables for input current decomposition plot in DCM
beta = 0.7 * np.pi
alpha = np.arctan((1 - np.cos(beta)) / (beta - np.sin(beta)))
gain = np.sin(alpha)
fn = 100

# Save the current decomposition to a csv file
current_directory = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_directory, "Grid_current_B2U_decomposition.csv")
np.savetxt(
    save_path,
    np.column_stack(
        (
            x,
            [i_1(xi, alpha, beta, gain) for xi in x],
            [i_1_fundamental(xi, alpha, beta, gain) for xi in x],
            [i_1_harmonic(xi, alpha, beta, gain) for xi in x],
        )
    ),
    delimiter=",",
    header="wt, i_1, i_1_fundamental, i_1_harmonic",
    comments="",
)


# calculate the power factor for beta = 0.05 ... pi (DCM)
beta = np.linspace(0.05, np.pi, 50)
alpha = np.arctan((1 - np.cos(beta)) / (beta - np.sin(beta)))
gain = np.sin(alpha)
I_out = 1 / 2 * ((1 - np.cos(beta)) / gain - gain * np.square(beta) / 2)

# allocate empoty i_1_THD and lamb variables according to the number of beta values
i_1_THD_val = np.zeros(len(beta))
lamb_val = np.zeros(len(beta))

for i in range(len(beta)):
    fourier = [
        fourier_coefficients(k, alpha[i], beta[i], gain[i]) for k in range(0, fn - 1)
    ]
    i_1_THD_val[i] = i_1_THD(fourier)
    lamb_val[i] = lamb(fourier)


# calculate the power factor for alpha_dash = alpha[-1] ... pi/2 (BCM)
alpha_dash = np.linspace(alpha[-1], np.pi / 2, 50)
gain_dash = 2 / np.pi * np.cos(alpha_dash)
I_out = np.concatenate((I_out, np.sin(alpha_dash)))

# extend the i_1_THD_val and lamb_val arrays with the new zero values for length of alpha_dash
i_1_THD_val = np.concatenate((i_1_THD_val, np.zeros(len(alpha_dash))))
lamb_val = np.concatenate((lamb_val, np.zeros(len(alpha_dash))))

for i in range(len(alpha_dash)):
    fourier = [
        fourier_coefficients_BCM(k, alpha_dash[i], gain_dash[i])
        for k in range(0, fn - 1)
    ]
    i_1_THD_val[i + len(beta)] = i_1_THD(fourier)
    lamb_val[i + len(beta)] = lamb(fourier)

# save the data to a csv file with I_out being the first column, i_1_THD_val the second, lamb_val the third to same folder as the script
save_path = os.path.join(current_directory, "Grid_impacts_B2U_rectifier.csv")
np.savetxt(
    save_path,
    np.column_stack((I_out, i_1_THD_val, lamb_val)),
    delimiter=",",
    header="I_out, i_1_THD, lambda",
    comments="",
)
