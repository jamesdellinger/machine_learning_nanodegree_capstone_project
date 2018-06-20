#Visualization module for Udacity Machine Learning Engineer
# Nanodegree Capstone Project

# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

# Import libraries necessary for this file.
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
# Set dpi of plots displayed inline
mpl.rcParams['figure.dpi'] = 100
# Configure style of plots
plt.style.use('fivethirtyeight')
# Make plots smaller
sns.set_context('paper')

def plot_feature_distributions(data, title, figsize, num_cols):
    """
    Plot distributions of a dataset's continuous features as histograms.

    Parameters:
        data: Pandas dataframe containing the features
        title: String, title of the figure
        figsize: Tuple, the dimensions in inches of the
                 figure that gets plotted.
        num_cols: Int, number of columns desired for the
                  figure
    """
    # Get list of dataframe's column names (the features)
    column_names = list(data.columns.values)

    # Get the number of features that will be plotted
    number_of_features = len(column_names)

    # Create a figure with 4 columns
    num_cols = num_cols

    # Ensure that there will be enough rows to
    # display each plot
    num_rows = int(np.ceil(number_of_features*1./num_cols))

    # Create the figure
    fig = plt.figure(dpi=300, figsize = figsize)

    # Plot the distribution of each feature
    for i, feature in enumerate(column_names):
        # Filter the feature's data for NaN values
        feature_data = data[feature]
        filtered_feature_data = feature_data[~np.isnan(feature_data)]
        # Add a new subplot for each feature, filling out a
        # grid that is num_rows x num_cols in dimensions.
        # Subplot index begins at 1.
        ax = fig.add_subplot(num_rows, num_cols, i+1)
        ax.hist(filtered_feature_data, bins = 25)
        ax.set_title("'%s' Distribution"%(feature), fontsize = 12)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Borrowers")

    # Plot aesthetics
    fig.suptitle(title, fontsize = 16, y = 1.03)

    # Display the plot
    fig.tight_layout()
    fig.show()

    # Save the plot to a png file
    fig.savefig('{}.png'.format(title))
