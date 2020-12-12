# ==============================================================================
# Script gathers all functionalities of impl/* scripts. Configuration variables
# should be set before running.
#
# @note : You should 'source' this script instead of executing. Source it
#    from the project's home directory.
#
# @note: If TF_VERSION=amd is used, the project's environment willbe run using
#    a prepacked docker container
# ==============================================================================

# Project's home directory
PROJECT_HOME=~/Desktop/SNR-Project

# Type of the tensorflow installation ['c': CPU, 'nvd': NVIDIA GPU, 'amd': AMD GPU]
TF_VERSION=amd

# Dataset to be downloaded from Kaggle (in form <owner>/<dataset-name>)
DATASET=moltean/fruits

# Name of the docker image (if AMD GPU used)
DOCK_IMG='snr-rocm'

# -------------------------------------- Exports -----------------------------------------------

export PROJECT_HOME=$PROJECT_HOME
export TF_VERSION=$TF_VERSION
export DATASET=$DATASET
export KAGGLE_CONFIG_DIR=$PROJECT_HOME/config/kaggle#
export DOCK_IMG=$DOCK_IMG


# ------------------------------------ Handy aliases -------------------------------------------

if [[ $TF_VERSION == "amd" ]]; then
    source $PROJECT_HOME/scripts/impl/aliases.bash
fi


# ------------------------------------Scripts calls --------------------------------------------

# Download data set
$PROJECT_HOME/scripts/impl/data.bash

# AMD GPU environment
if [[ $TF_VERSION == "amd" ]]; then

    # Install docker
    $PROJECT_HOME/scripts/impl/docker_install.bash

    # Build the image and run the container
    source $PROJECT_HOME/scripts/impl/rocm.bash

# CPU / Nvidia GPU environment
elif [[ $TF_VERSION == "c" || $TF_VERSION == "nvd" ]]; then

    # Install dependancies
    source $PROJECT_HOME/scripts/impl/installs.bash

    # Setup the environment
    source $PROJECT_HOME/scripts/impl/venv.bash    

else
    echo "ERR: Wrong tensorflow version ('$TF_VERSION=c' unknown)"
fi
