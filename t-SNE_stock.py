# Version 2.0 of the t-SNE Stock Market Example: with triggers and better defined functions.
# Try with simulated data so we know that they are clustered 

#___________________________________________________________________________________________________ Import packages

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import colors
from fbm import FBM

# Use the normalization? Shouldn't be needed
# from sklearn.preprocessing import normalize
import imageio
from timeit import default_timer as timer
import umap

#___________________________________________________________________________________________________ Control triggers

####################### Triggers ########################
# 0: no, 1: yes
plot_bar        = 0 # Bar plots of returns for all companies for a given number of days (randomized)
compute_tsne    = 0
plot_tsne_synth = 0 # Plots a single plot of t-SNE for the snythetic dataset 
plot_tsne_gif   = 0 # Plots a gif of all the possible windows of t-SNE
compute_umap    = 0
plot_umap       = 0

###################### Parameters ########################

window_time     = 40 # Window of time for the "dynamic t-SNE"
days_bar_plot   = 3  # Number of bar plots (also number of days it will be plotted for)


#___________________________________________________________________________________________________ Import data

raw_data = np.asarray(pd.read_csv('all_stocks_5yr.csv')) # Extract data from the example dataset

# Extract oder parameters from the dataset
company_name        = list([line.rstrip('\n') for line in open('wholenames.txt')])
categories          = list([line.rstrip('\n') for line in open('category.txt')])
unique_categories   = np.unique(categories)
acronym             = np.unique(raw_data[:,6])

#___________________________________________________________________________________________________ Extract days with 
#                                                                                                    containing all data

# Function to extract all the possible days 
def extract_days(raw_data):
    """ Extracts all the days that have information regarding all 500 companies. """
    
    # Extract necessary data from the input dataset
    unique_days                 = np.unique(raw_data[:,0])
    number_total_companies      = len(np.unique(raw_data[:,6]))
    days_all_companies          = []
    
    for i in range(len(unique_days)):
        companies_given_days = len(raw_data[raw_data[:,0] == unique_days[i]])
        # We only want the days that have information regarding all companies of interest.
        if companies_given_days != number_total_companies:
            pass
        else:
            days_all_companies.append(unique_days[i])
            
    return np.asarray(days_all_companies)

# Extract the complete weeks
complete_days = extract_days(raw_data)

#___________________________________________________________________________________________________ Calculate the logarithmic 
#                                                                                                    returns

def log_returns(raw_data, days):
    """ Calculates the logarithmic returns per company for each of the set of days. """
    
    # 1.- Separate all the data by the opening and returns for each corresponding day.
    day_data = np.zeros((len(days), len(company_name), 4)) 
    
    # The 4 is hardcoded and should be changed. It accounts for the 4 columns of data
    # that do not correspond to the day and to the company acronym. 
    
    for i in range(len(days)):
        day_data[i] = (raw_data[raw_data[:,0] == days[i]])[:,1:5]
        
    # 2.- Calculate the logarithmic returns for each day: 
    # log(1 + returns) takes care of the invalid values and we assume the returns are norm. 
    returns = np.zeros((len(days), len(company_name)))
    for i in range(len(days)):
        opening     = day_data[i][:,0]
        closing     = day_data[i][:,3]
        returns[i]  = np.log(1 + (closing-opening)/opening) 
    
    return returns, day_data

returns_all, day_data = log_returns(raw_data, complete_days)
labels_all = np.full(len(company_name), 0)
open_all    = day_data[:,:,0]
close_all   = day_data[:,:,3]

# Plot the closing data of the real companies to compare with the synthetic 
# Plots all companies for one day
#plt.figure()
#for i in range(len(company_name)):
#    plt.plot(np.linspace(0, 1, len(complete_days)), close_all[:,i], alpha=0.35, c='black')
#    

#___________________________________________________________________________________________________ Snythetic dataset of returns

# REMINDER THAT THE ABOVE THAT SET IS RELATED TO THE LOGARITHMIC RETURNS AND NOT TO THE PRICES 
# CHANGE SIMULATION 

# Create linear growth and linear loss, stagnated around 0 and random motions with Browninan sims.
# Define general parameters for all the synthesis:
    # Define the number of days we are simulating
    # And the number of points aka companies 
    
sim_days = len(complete_days)
sim_comp = 150 # Companies with a given characteristic
min_ = -0.07 # Minimum returns 
max_ = 0.05  # Max logarithmic returns

# 1.- Linear dataset, noise has to be added at the end

def linear_(simulated_days, simulated_companies, max_, min_):
    """ Top refers to the end of the array, whether it is ascending or descending. """
    np.random.seed(0)
    
    # Create the different arrays for the linear growth
    linear_growth_ = []
    for i in range(simulated_companies):
        arr_g = np.linspace(0, np.random.normal(max_, max_/10), simulated_days)
        linear_growth_.append(arr_g)
        
    # Create the different arrays for the linear loss
    linear_loss_   = []
    for i in range(simulated_companies):
        arr_l = np.linspace(0, np.random.normal(-max_, -min_/10), simulated_days)
        linear_loss_.append(arr_l)
        
    return np.asarray(linear_growth_), np.asarray(linear_loss_)


linear_growth_, linear_loss_ = linear_(sim_days, sim_comp, max_, min_)

# Add Gaussian noise to each of the above arrays

linear_growth   = linear_growth_ + np.random.normal(0, max_/10, (linear_growth_.shape))
lingrow_labels  = np.full(sim_comp, 1)
linear_loss     = linear_loss_ + np.random.normal(0, max_/10, (linear_loss_.shape))
linloss_labels  = np.full(sim_comp, 2)

# 2.- Define the companies with a mean of 0 returns
flat            = np.zeros((linear_growth.shape)) + np.random.normal(0, max_/3.5, (linear_loss_.shape))
flat_labels = np.full(sim_comp, 3)

# 3.- Brownian motion 
def brownian_(simulated_days, simulated_companies):
    """ Creates Brownian motions to simulate reallistical stock market data """
    np.random.seed(0)
    # Define a list of possible incremental steps to give some variability to the dataset
    
    increments = np.random.beta(10, 0.75, 10000)/150 # The divisor value scales the values to the
                                                     # previously defined maximum 
    brownian = []
    for i in range(simulated_companies):
        # Shuffle the increment value
        np.random.shuffle(increments)
        # Calculate the brownian motion per-se
        brownian_ = np.cumsum(np.random.normal(0, increments[i], simulated_days))
        # Append the results
        brownian.append(brownian_)
    return np.asarray(brownian)

brownian        = brownian_(sim_days, sim_comp*2)
brownian_labels = np.full(len(brownian[:,]), 4)

#plt.figure()
#for i in range(150):
#    plt.plot(np.linspace(0, 1, sim_days), flat[i], alpha=0.15, c='blue')
#    plt.plot(np.linspace(0, 1, sim_days), linear_growth[i], alpha=0.15, c='green')
#    plt.plot(np.linspace(0, 1, sim_days), linear_loss[i], alpha=0.15, c='red')
#for i in range(300):
#    plt.plot(np.linspace(0, 1, sim_days), brownian[i], alpha=0.1, c='black')
#for i in range(len(company_name)):
#    plt.plot(np.linspace(0, 1, len(complete_days)), returns_all[:,i], alpha=0.1, c='pink')

# Put all the data together in a single array wit the shape (days, companies)
returns_all_synth = np.concatenate([linear_growth, linear_loss, flat, brownian])
labels_all_synth  = np.concatenate([lingrow_labels, linloss_labels, flat_labels, brownian_labels])

ALL_returns = np.concatenate([returns_all.T, returns_all_synth])
ALL_labels  = np.concatenate([labels_all, labels_all_synth])

#___________________________________________________________________________________________________ Snythetic dataset of returns 2.0
#                                                                                                    using a more realistic model

# Fractional Brownian Motion - continuous time Gaussian Process 
# Calculate the LOGARITHMIC returns from the fractional brownian motions 
# ALSO divide by opening value for normalization 

def synth_returns_fbm(sim_days, sim_comp):    
    """ Calculates the logarithmic returns for stock data simulated using random
    walks (fractional brownian motion). """
    hursts = np.linspace(0.45, 0.75, 100)
    returns_synth = []
    for i in range(sim_comp):          
        np.random.shuffle(hursts)
        f = FBM(n=sim_days, hurst=hursts[0], length=1, method='daviesharte')
        fbm_sample = f.fbm()
        
        # Calculate the logarithmic returns
        returns_ = fbm_sample
        returns_synth.append(returns_)
        
    return np.asarray(returns_synth)


returns_fbm = synth_returns_fbm(sim_days, 505)
fbm_labels = np.full((len(returns_fbm)), 6)


returns_FBM_REAL = np.concatenate([returns_fbm/4, returns_all.T], axis=0)
labels_FBM_REAL  = np.concatenate([fbm_labels, labels_all])


# Print to check
plt.figure()
for i in range(len(company_name)):
    plt.plot(np.linspace(0, 1, len(complete_days)), returns_all[:,i], alpha=0.05, c='black')
    plt.plot(np.linspace(0, 1, 44), returns_fbm[i]/5, alpha=0.05, c='pink')
    
#___________________________________________________________________________________________________ Plot of single t-SNE with 
#                                                                                                    synthetic data
if plot_tsne_synth == 1:
    # Plots a single projection of t-SNE 
    # Ask for user input 
#    window_plot_single_tsne = int(input())
#    print("Selected window between days {} and {}".format(complete_days[window_plot_single_tsne],
#               complete_days[window_plot_single_tsne + window_time]))
    
    
    # Calculate t-SNE for the synthetic dataset 
    tsned_data_synth =  TSNE(perplexity=10, random_state=0, init='pca', early_exaggeration=50).fit_transform(returns_FBM_REAL)
    tsned_data_synth = umap.UMAP(n_neighbors=50, min_dist=0.00, n_components=2, metric='euclidean').fit_transform(returns_all_synth)
    
    x = tsned_data_synth[:,0]
    y = tsned_data_synth[:,1]
    
    x_real = x[0:len(company_name)]
    y_real = y[0:len(company_name)]
    
    x_ling = x[len(company_name):len(company_name) + sim_days]
    y_ling = y[len(company_name):len(company_name) + sim_days]
    
    x_lind = x[len(company_name) + sim_days:len(company_name) + 2*sim_days]
    y_lind = y[len(company_name) + sim_days:len(company_name) + 2*sim_days]
    
    x_flat = x[len(company_name) + 2*sim_days:len(company_name) + 3*sim_days]
    y_flat = y[len(company_name) + 2*sim_days:len(company_name) + 3*sim_days]
    
    x_brown = x[-len(brownian):]
    y_brown = y[-len(brownian):]

    mean_returns_synth = np.mean(returns_FBM_REAL, axis=1)
    
    # Plot the single t-SNE projection
    fig, ax = plt.subplots(figsize=(10,10))  
    ax.scatter(x_real, y_real, s=5, label='Real')
    ax.scatter(x_ling, y_ling, s=5, label='Grow')
    ax.scatter(x_lind, y_lind, s=5, label='Loss')
    ax.scatter(x_flat, y_flat, s=5, label='Flat')
    ax.scatter(x_brown, y_brown, s=5, label='Brown')
    plt.legend()
    
    x_fbm = x[0:len(returns_fbm)]
    y_fbm = y[0:len(returns_fbm)]
    
    x_real = x[-len(company_name):]
    y_real = y[-len(company_name):]
    
    fig, ax = plt.subplots(figsize=(10,10))  
    ax.scatter(x_real, y_real, s=5, label='Real')
    ax.scatter(x_fbm, y_fbm, s=5, label='FBM')
    plt.legend()
    
    fig, ax = plt.subplots(figsize=(10,10))  
    p = ax.scatter(x, y, s=5, c=labels_FBM_REAL)





#___________________________________________________________________________________________________ 
#___________________________________________________________________________________________________  
#___________________________________________________________________________________________________ 
#___________________________________________________________________________________________________ 
#___________________________________________________________________________________________________ 
#___________________________________________________________________________________________________ t-SNE calculations for the 
#                                                                                                    real dataset. 

if compute_tsne == 1:
    
    # Append to store all the information, we need a variable size, depends on the selected
    # window 
    tsne_data = []
    day_tsne_data = []
    
    # Calculates t-SNE for different groups of days to see the change in behavior
    s_time = timer()
    for i in range(len(complete_days)):
        if (i + window_time) < len(complete_days):
            # Performs the t-SNE calculations for the given set/window of days. Initialized with
            # PCA.          
            tsne_range = TSNE(perplexity=200, random_state=0, init='pca',
                              early_exaggeration=50).fit_transform(returns_all[i:i + window_time].T)
            
            day_tsne_range  = (complete_days[i], complete_days[i + window_time])          
            tsne_data.append(tsne_range)
            day_tsne_data.append(day_tsne_range)
            
            # To follow the progress of the calculations
            print(i)
            
        else: # To stop the loop when the above condition fails
            break
        
    e_time = timer()
    
    print('t-SNE took {} seconds to finish the {} calculations'.format((e_time - s_time),
          len(tsne_data)))

#___________________________________________________________________________________________________ Plot GIF of
#                                                                                                    dynamic t-SNE

if plot_tsne_gif == 1:
    
    def plot_tsne(tsne_data, returns_total, i):
        """ Plots tSNE for the time interval given in window_time. """
        
        tsned_data = np.asarray(tsne_data)
        x = tsned_data[i][:,0]
        y = tsned_data[i][:,1]
        mean_returns = np.mean(returns_all[i:i + window_time,:], axis=0)
        
        # Create labels for gain or loss
        labels_behaviour = np.zeros(len(tsned_data[0,:]))
        
        for j in range(len(tsned_data[0,:])):
            if mean_returns[j] > 0:
                labels_behaviour[j] = 1
            if mean_returns[j] < 0:
                labels_behaviour[j] = -1

#        # Set the colorbar
#        bounds = np.linspace(returns_all.min(), returns_all.max(), len(tsned_data[0,:]))
#        cmaps  = colors.ListedColormap('RdYlGn')
#        norms  = colors.BoundaryNorm(bounds, cmaps.N)
        
        # Actual plotting a
        # Maybe even add clustering?
        fig, ax = plt.subplots(figsize=(10,10))
        
        ax.scatter(x, y, c=mean_returns, cmap='RdYlGn')
        
        ax.set(title='Between days {} and {}'.format(complete_days[i],
               complete_days[i + window_time]))

        # Animation code
        # Used to keep the limits constant
        ax.set_ylim(-45, 45)
        ax.set_xlim(-45, 45)
    
        # Used to return the plot as an image rray
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return image
    
    kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    imageio.mimsave('./tsne.gif',
                    [plot_tsne(tsne_data, returns_all,
                               i) for i in range(len(complete_days) - window_time)], fps=2)    
    
#___________________________________________________________________________________________________ UMAP Test
    
if compute_umap == 1:
    umap_data     = []
    day_umap_data = []
    
    # Calculates t-SNE for different groups of days to see the change in behavior
    s_time = timer()
    for i in range(len(complete_days)):
        if (i + window_time) < len(complete_days):
            # Performs the t-SNE calculations for the given set/window of days. Initialized with
            # PCA.
            
            umap_range = umap.UMAP(n_neighbors=50, min_dist=0.00, n_components=2, 
                                   metric='euclidean').fit_transform(returns_all[i:i + window_time].T)
            
            day_umap_range  = (complete_days[i], complete_days[i + window_time]) # Rewrite as a touple 
            
            umap_data.append(umap_range)
            day_umap_data.append(day_umap_range)
            
            # To follow the progress of the calculations
            print(i)
            
        else: # To stop the loop when the above condition fails
            break
        
    e_time = timer()
    
    print('UMAP took {} seconds to finish the {} calculations'.format((e_time - s_time),
          len(umap_data)))
#___________________________________________________________________________________________________ UMAP Plot GIF

    
if plot_umap == 1:
    
    def plot_umap(umap_data, returns_total, i):
        """ Plots tSNE for the time interval given in window_time. """
        
        umapd_data = np.asarray(umap_data)
        x = umapd_data[i][:,0]
        y = umapd_data[i][:,1]
        mean_returns = np.mean(returns_all[i:i + window_time,:], axis=0)
        
        # Create labels for gain or loss
        labels_behaviour = np.zeros(len(umapd_data[0,:]))
        
        for j in range(len(umapd_data[0,:])):
            if mean_returns[j] > 0:
                labels_behaviour[j] = 1
            if mean_returns[j] < 0:
                labels_behaviour[j] = -1

#        # Set the colorbar
#        bounds = np.linspace(returns_all.min(), returns_all.max(), len(tsned_data[0,:]))
#        cmaps  = colors.ListedColormap('RdYlGn')
#        norms  = colors.BoundaryNorm(bounds, cmaps.N)
        
        # Actual plotting 
        # Maybe even add clustering?
        fig, ax = plt.subplots(figsize=(10,10))
        
        ax.scatter(x, y, c=mean_returns, cmap='RdYlGn')
        
        ax.set(title='Between days {} and {}'.format(complete_days[i],
               complete_days[i + window_time]))

        # Animation code
        # Used to keep the limits constant

        # Used to return the plot as an image rray
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return image
    
    kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    imageio.mimsave('./umap_{}days.gif'.format(window_time),
                    [plot_umap(umap_data, returns_all,
                               i) for i in range(len(complete_days) - window_time)], fps=1) 
    
#___________________________________________________________________________________________________ Calculate and print 
#                                                                                                    the correlation matrix

# Calculate and plot the correlation matrix for a given day

def correlation_matrix(returns, company_names):
    """ Calculates and plots the correlation matrix for the dataset. """
    # 1.- Transform the data into an appropiate dataframe.
    
    df      = pd.DataFrame(returns) # Should me all the returns.
    corr    = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
#___________________________________________________________________________________________________  Bar plot of the returns for 
#                                                                                                    each company for a given day

if plot_bar == 1:
    
    # Selects a predefined number of random days for the bar plots.
    np.random.seed(0)
    randomized_days = (np.random.permutation(len(complete_days)))[0:days_bar_plot]
    
    for day_ in randomized_days:     
        
       plt.figure(figsize=(10,4))
       
       # Define the colormap for a more organized color plot
       cmap = plt.cm.RdYlGn
       norm = colors.Normalize(vmin=min(returns_all[day_,:]),
                               vmax=max(returns_all[day_,:]))
       
       # Proceed with the actual plotting 
       plt.bar(acronym, returns_all[day_,:], width=1.25, color=cmap(norm(returns_all[day_,:])))
       plt.title('Logarithmic returns S&P500 on the {}'.format(complete_days[day_]))
       plt.ylabel('Logarithmic returns (x100)')
       plt.xticks(rotation=45)
    