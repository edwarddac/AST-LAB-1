# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.special



#--------------------------------------
#Question1
#Plots-start
xdata = np.arange(1,31)
ydata = [38.91, 37.14, 38.19, 41.03, 34.86, 37.33, 35.16, 37.96, 36.93, 40.41, 29.50, 37.33, 41.84, 37.53, 34.12, 34.11, 37.94, 34.43, 36.68, 41.31, 39.61, 35.48, 34.98, 39.05, 39.62, 37.96, 39.02, 37.47, 33.76, 36.51]

plt.plot(xdata, ydata, 'o')
plt.title('Distribution of the 30 Obtained Distances', fontsize=15)
plt.xlabel('Measurements', fontsize=15)
plt.ylabel('Distance (pc)',fontsize=15)
plt.grid()
plt.show()

np.std(ydata)
np.mean(ydata)

n, bins, patches = plt.hist(ydata, bins=[26, 30, 32.5, 35.5, 38.5, 41.5, 44.5, 47.5], facecolor='blue', alpha=0.5, ec='black')
plt.title('Historgram of the 30 Distance Measurments', fontsize=15)
plt.xlabel('Distance (pc)', fontsize=15)
plt.ylabel('Number of Measurements', fontsize=15)
plt.show()

#Plots-end


Dis_Msrm_with_error = [38.91, 1.41, 37.14, 0.36, 38.19, 0.69, 41.03, 3.53, 34.86, 2.64, \
            37.33, 0.17, 35.16, 2.34,37.96, 0.46, 36.93, 0.57, 40.41, 2.91, \
            29.50, 8.00, 37.33, 0.17, 41.84, 4.34, 37.53, 0.03, 34.12, 3.38, \
            34.11, 3.39, 37.94, 0.44, 34.43, 3.07, 36.68, 0.82, 41.31, 3.81, \
            39.61, 2.11, 35.48, 2.02, 34.98, 2.52, 39.05, 1.55, 39.62, 2.12, \
            37.96, 0.46, 39.02, 1.52, 37.47, 0.03, 33.76, 3.74, 36.51, 0.99]

Dis_Msrm = Dis_Msrm_with_error[::2]
Dis_Msrm_error = Dis_Msrm_with_error[1::2]

print('The values are '+ str(Dis_Msrm) + " in pc")

print('The errors associated with each value are ' + str(Dis_Msrm_error) + " in pc")


def mean(lst_of_values):
    j = 0
    for i in lst_of_values:
        j = j + i
    
    return j/len(lst_of_values)

print("The mean of the values are " + str(mean(Dis_Msrm))+ "pc")

def std(lst_of_values):
    j = 0
    for value in lst_of_values:
        j = j + abs(value - mean(lst_of_values))**2
    return math.sqrt(j/(len(lst_of_values)-1))

print("The error of the mean is " + str(std(Dis_Msrm))+ "pc") 

def weight(lst_of_errors):
    """ 
    Return a list of weights of the values with errors from 
    lst_of_errors.
    """
    lst_of_weights = []
    for value in lst_of_errors:
        lst_of_weights.append(1/(value**2))
    return lst_of_weights

print("The weights of each value are " + str(weight(Dis_Msrm_error))) 

def weighted_mean(lst_of_weights, lst_of_values):
    i = 0
    for value in lst_of_values:
        i = i + value*lst_of_weights[lst_of_values.index(value)]
    return i/sum(lst_of_weights)

print("The weighted mean is " + str(weighted_mean(weight(Dis_Msrm_error), Dis_Msrm))+ "pc")

def std_of_wmean(lst_of_errors):
    i = 0
    for error in lst_of_errors:
        i = i + 1/(error**2)
    return math.sqrt(1/i)

print("The error in the weighted mean is " + str(std_of_wmean(Dis_Msrm_error)) + "pc")

#--------------------------------------
#Question2

lst_of_photon_count_rate = [13, 17, 18, 14, 11, 8, 21, 18, 9, 12, 9, 17, 14, 6,
                            10, 16, 16, 11, 10, 12, 8, 20, 14, 10, 14, 17, 13, 
                            16, 12, 10]

print("The mean of the Photon Count Rate is " + str(mean(lst_of_photon_count_rate)) + " counts/sec.")
print("The error in the mean of the Photon Count Rate is " + str(std(lst_of_photon_count_rate))+ " counts/sec.")
 

def P(x,u):
    f1 = u**x/scipy.special.factorial(x)
    f2 = np.exp(-u)
    return (f1*f2)


#Plot
xdata2 = np.arange(5,24, 0.01)
n, bins, patches = plt.hist(lst_of_photon_count_rate, bins =[5, 7, 9, 11, 13, 15, 17, 19, 22, 25], facecolor='g', alpha=0.5, normed = False, label="Photon Count Rate")
plt.title('Histogram of the photon count rate', fontsize=15)
plt.xlabel('Number of Photons', fontsize=15)
plt.ylabel('Number of Measurements', fontsize=15)
plt.plot(xdata2, P(xdata2, 12)*61.2059, 'y--', label = 'Expected Poisson distribution with u = 12')
plt.legend(loc=4)
plt.show()

