# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2021-01-27 09:03:21
# @ Modified time: 2021-01-27 09:08:18
# @ Description:
#
#     Script draws val_accuracy charts for all 'run_1' trainings' of every model ('run_1' always contains
#     a reference run).
#
# ================================================================================================================

# Name of the output file
outname = 'val_loss.pdf'

# Metric to be plotted
metric = 'val_accuracy'

# Plot's title
title = 'Learning rates'

# Legend's location
lgd_loc = 'lower right'

# ----------------------------------------------------------------------------------------------------------------

import os
import pickle
from matplotlib import pyplot as plt

# Get path to the models' directory
home = os.getenv('PROJECT_HOME')
models = os.path.join(home, 'models')

# Prepare figure
fig, ax = plt.subplots(figsize=(5,3))
ax.set_title(title)
ax.set_ylabel(metric)
ax.set_xlabel('epoch')

# Set log scale
plt.yscale('log')

# Iterate over 'run_1' folders in all models 
for model in os.listdir(models):

    # Check whether 'run_1' subfolder exists
    datadir = os.path.join(os.path.join(models, model), 'run_1')
    if not os.path.exists(datadir):
        continue

    # Load training history
    with open(os.path.join(datadir, 'history/subrun_1.pickle'), 'rb') as h:
        history = pickle.load(h)

    # Append metric to the list
    plt.plot(history[metric])

# Set legend
lgnd = ax.legend(os.listdir(models), loc=lgd_loc, shadow=True)

# Print figure
fig.savefig(os.path.join(os.path.join(home, 'visualization/data'), outname), format='pdf')
plt.show()