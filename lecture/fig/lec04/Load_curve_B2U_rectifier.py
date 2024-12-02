# Calculate B2U rectifier load curve and save it to a csv file
# (Required math operations resulted in mathtikz errors, so numpy was used instead)

import numpy as np
import os

# define variable beta = 0...pi with 100 points
beta = np.linspace(0.01, np.pi, 100)
alpha = np.arctan((1-np.cos(beta))/(beta-np.sin(beta)))
gain = np.sin(alpha)
I_out = 1/2 * ((1-np.cos(beta))/gain - gain*np.square(beta)/2)

# define alpha_dash from the last value of alpha to pi/2 with additional 100 points
alpha_dash = np.linspace(alpha[-1], np.pi/2, 100)

# Concatenate the new values to the existing arrays
gain = np.concatenate((gain, 2/np.pi * np.cos(alpha_dash)))
I_out = np.concatenate((I_out, np.sin(alpha_dash)))
alpha = np.concatenate((alpha, alpha_dash))

# add ones to the end of beta array to match the length of gain
beta = np.concatenate((beta, np.ones(100)*np.pi))

# save the data to a csv filw with I_out being the first column, gain the second, alpha the third and beta the fourth to same folder as the script
current_directory = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_directory, 'Load_curve_B2U_rectifier.csv')
np.savetxt(save_path, np.column_stack((I_out, gain, alpha/np.pi, beta/np.pi)), delimiter=',', header='I_out, gain, alpha, beta', comments='')

