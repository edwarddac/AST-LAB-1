""" code """
import numpy as np
import matplotlib.pylab as plt

filename = 'dist_data.txt'
dist, dist_err = np.loadtxt(filename, skiprows=6, unpack=True)


# CALCULATE MEAN AND STANDARD DEVIATION


def calculate_mean_and_std(values):
    """
    Return the Mean and "Sample" Standard Deviation given an array of measurements.

    We use the 'Sample Standard Deviation' because we are using a sample, not the whole population.

    :param values: A collection of measurements - numpy array
    :return: Mean - float, Standard Deviation - float
    """
    average = sum(values)/(len(values))
    return average, np.sqrt(sum((values - average)**2)/(len(values) - 1))


scratch_mean, scratch_STD = calculate_mean_and_std(dist)
print("\nScratch Mean and STD are:", scratch_mean, scratch_STD)

# Check work using builtin functions
dist_mean = np.mean(dist)
dist_std = np.std(dist)

print('\nBuilt-in Mean and STD are:', dist_mean, dist_std)


# CALCULATE WEIGHTED AVERAGES


def calculate_weighted_mean_and_std(values_arr, error_arr):
    """
    Return the weighted mean and standard deviation given an array of values and associated list of errors

    :param values_arr: a numpy array of measurements
    :param error_arr: a numpy array of errors associated with the measurements
    :return: Weigh
    """
    weights = 1.0/(error_arr**2)
    return sum(values_arr * weights)/sum(weights), np.sqrt(1.0/sum(weights))


scratch_weighted_mean, scratch_weighted_STD = calculate_weighted_mean_and_std(dist, dist_err)
print("Scratch Weighted Mean and STD", scratch_weighted_mean, scratch_weighted_STD)
print()


# doing it with built in functions


def weighted_avg_and_std(values, errors):
    """   """
    weights = 1.0 / (errors ** 2)
    weights_normalized = np.array([float(i) / sum(weights) for i in weights])
    average = np.average(values, weights=weights_normalized)
    variance = np.average((values - average)**2, weights=weights_normalized)
    return average, np.sqrt(variance)


avg_BI, std_BI = weighted_avg_and_std(dist, dist_err)
print("Weighted Avg and STD:", avg_BI, std_BI)

# PLOT figures:
x = np.arange(1, 31)
plt.scatter(x, dist)
plt.xlabel('Measurement')
plt.ylabel("Distance (pc)")
plt.show()

plt.hist(dist, bins=[27, 30, 33, 36, 38, 41, 44])
plt.xlabel("Distance (pc)")
plt.ylabel("Number of Measurements")
plt.show()

# ~~~~~~~~ SECTION 4: POISSION DISTRIBUTION AND TIME SERIES DATA ~~~~~~~~~~~~~
print("\n~~~~~~~~ SECTION 4: POISSION DISTRIBUTION AND TIME SERIES DATA ~~~~~~~~~~~~\n")
photon_rate = np.loadtxt('photon_count_rate.txt', skiprows=3, unpack=True)
photon_mean, photon_std = calculate_mean_and_std(photon_rate)
print("Photon mean and std are:", photon_mean, photon_std)

plt.hist(photon_rate, bins=[5, 7, 9, 11, 13, 15, 17, 19, 21, 23])
plt.xlabel("Number of Photons")
plt.ylabel("Number of Measurements")
x_pho = np.linspace(6, 22)
mu = 12.0
y_pho = (mu**x_pho)*np.exp(-12.0)/np.factorial(x_pho)
plt.plot(x_pho, y_pho)
plt.show()