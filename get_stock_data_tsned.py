# Download stock market data.
#___________________________________________________________________________________________________ Import packages

import pandas_datareader.data as web
import pandas as pd
import datetime
import numpy as np
from timeit import default_timer as timer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import imageio
import time 

#___________________________________________________________________________________________________ Triggers

compute_tsne    = 0
plot_tsne_gif   = 0


#___________________________________________________________________________________________________ Import data

# Import the data - obtain company names and then compare with the raw data we already had
sp_names    = pd.read_excel('SP500.xlsx')
symbols     = sp_names['Symbol'] 

# Solve the problem with BRK.B removing the last B, as it is causing the problem
symbols[72] = 'BRK'
symbols[85] = 'BF'


#___________________________________________________________________________________________________ Input parameters
 
start       = datetime.datetime(2010,1,1)
end         = datetime.datetime(2019,8,6)
provider    = 'yahoo'

window_time = 50

#
#___________________________________________________________________________________________________ Obtain data for the 
#                                                                                                    SP500

# Retrueve raw data for the SP500
raw_SP500 = []
errors = []
for i in range(len(symbols)):
    if i != 0 and i % 100 == 0:
        # In order to avoiid hitting the limit for max requests at once
        time.sleep(120)
    try:
        time.sleep(10)
        raw_SP500.append(web.DataReader(symbols[i], 'yahoo', start, end))
        print(i)
    except KeyError:
        print('Key error happened at {}'.format(i))
        errors.append(i)

#___________________________________________________________________________________________________ Obtain data and calculate
#                                                                                                    the logarithmic returns

def log_returns_query():
    s_time = timer()
    # Retrieve the data before the calculations
    print('Retrieving stock data between {} and {} from {}'.format(start, end, provider))
    raw = web.DataReader(symbols, provider, start, end)
    e_time = timer()
    print('Data retrieving finished, it took {} seconds'.format((e_time - s_time)))

    # The input data should come directly from the query.
    # Build query inside this function? 
    
    opening = raw['Open']
    closing = raw['Close']
    
    # Calculate the actual log returns, handle the nan as zeros?
    factor      = np.nan_to_num((closing - opening)/opening)
    log_returns = np.log(1 + factor)
    
    # All the exports ara dataframes
    return log_returns

returns_all = log_returns_query()

#___________________________________________________________________________________________________ t-SNE calculations (for DAX)

if compute_tsne == 1:
    
    # Append to store all the information, we need a variable size, depends on the selected
    # window 
    tsne_data = []
    day_tsne_data = []
    
    # Calculates t-SNE for different groups of days to see the change in behavior
    s_time = timer()
    for i in range(len(days)):
        if (i + window_time) < len(days):
            # Performs the t-SNE calculations for the given set/window of days. Initialized with
            # PCA.
            
            tsne_range = TSNE(perplexity=30, random_state=0, init='pca',
                         early_exaggeration=50).fit_transform(returns_all[i:i + window_time].T)
            
            day_umap_range  = (days[i], days[i + window_time])
            
            tsne_data.append(tsne_range)
            day_tsne_data.append(day_umap_range)
            
            # To follow the progress of the calculations
            print(i)
            
        else: # To stop the loop when the above condition fails
            break
        
    e_time = timer()
    
    print('t-SNE took {} seconds to finish the {} calculations'.format((e_time - s_time),
          len(tsne_data)))


#___________________________________________________________________________________________________ Plot GIF of
#                                                                                                    t-SNE (of the DAX)

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
        
        # Actual plotting 
        # Maybe even add clustering?
        fig, ax = plt.subplots(figsize=(10,10))
        
        ax.scatter(x, y, c=mean_returns, cmap='RdYlGn')
        
        ax.set(title='Between days {} and {}'.format(days[i],
               days[i + window_time]))

        # Animation code
        # Used to keep the limits constant
#        ax.set_ylim(-25, 25)
#        ax.set_xlim(-25, 25)
    
        # Used to return the plot as an image rray
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return image
    
    kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    imageio.mimsave('./tsne{}daywindow_yahooquery.gif'.format(window_time),
                    [plot_tsne(tsne_data, returns_all,
                               i) for i in range(len(tsne_data))], fps=1)    
