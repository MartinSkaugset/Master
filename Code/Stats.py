# -*- coding: utf-8 -*-
"""
Created on Sun Jun 6 17:10:00 2021

@author: Martin Skaugset
"""
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt

my_path = os.path.dirname(os.path.abspath('simulator.py'))

def statistcal_analysis(data):
    data_vec = []
    mean = np.mean(data)
    median = np.median(data)
    standard_deviation = np.std(data)
    variance = standard_deviation**2
    std_mean_error = stats.sem(data)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    max_data = max(data)
    min_data = min(data)
    range_width = max_data - min_data
    coeff_variability = standard_deviation/mean
    data_vec.extend([mean, median, standard_deviation, variance,std_mean_error, skewness, 
                     kurtosis, max_data, min_data, range_width,coeff_variability])
    return data_vec


if __name__ == '__main__':

    raw_data = np.loadtxt(str(my_path + '/Textfiles/Results/50_percentile_09_test.txt'))

    #raw_data = np.loadtxt(str(my_path + '/Textfiles/Results/R2/R2_09_tot.txt'))
    data = statistcal_analysis(raw_data)
    tot_mean = np.zeros(len(raw_data))
    for i in range(len(raw_data)):
        tot_mean[i] = np.mean(raw_data[0:i])
        
    plot_length = np.arange(len(raw_data))
    
    plt.figure(1)
    plt.title("Mean of above 90th percentile with $θ_{1} = 0.9, θ_{2} = 0.1$")
    #plt.title("Mean of $R^2$ with $θ_{1} = 0.9, θ_{2} = 0.1$")
    plt.xlabel("Number of independent simulation")
    plt.ylabel("Mean value")
    plt.plot(plot_length,tot_mean)
    #plt.savefig(str(my_path + '/Figures/Results/90percentile/90percentile_09.jpg'),dpi = 600,format = 'jpg')
    #plt.savefig(str(my_path + '/Figures/Results/R2/R2_9.jpg'),dpi = 600,format = 'jpg')
    
    print("Length of data: ", len(raw_data))
    print("Mean:" , data[0])
    print("Median:" , data[1])
    print("Standard deviation: ", data[2])
    print("Standard mean error:" , data[4])
    print("Max:" , data[7])
    print("Min: ", data[8])