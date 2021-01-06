# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2020-11-11 13:40:58
# @ Modified time: 2020-12-20 17:40:31
# @ Description:
#     
#     Script gathers all functionalities of impl/* scripts. Configuration variables, defined under this header, 
#     should be set appropriately before running. After running the script the whole environment for working with
#     the project will be prepared. Lacking tools and python packages will be installed and the environment variables
#     will be set. Console used to source this script will be ready to use project's python scripts.
#    
#     In cases when a CPU/Nvidia GPU is used, the console sourcing the script will be switched to the python virtual
#     environment. Thanks to it, the user space will be not cluttered with additional python packages required by the
#     project.
#    
#     In case of AMD GPU, the script will run a docker container with a fully-isolated environment used to play
#     with project's code. The virtual machine will have the project's folder mounted and the user will be 
#     automatically switched to it so that they can instantly run the training. 
#    
# @ Note: Read 'config/kaggle/README.md' before sourcing this file
#
# @ Note: You should 'source' this script instead of executing. For future compatibility source it from the
#     project's home directory.
#
# @ Warning: sourcing this script will install lacking tools with sudo, without asking user at the every 
#     installation. Before sourcing the script, read content of the scripts/impl/*.bash files or source
#     this script in an isolated environment.
#
# @ Note: The script assumes that `python3` will be used to run project's code and all python packages are
#     installed using `python3 -m pip install`
# ================================================================================================================

# Project's home directory
PROJECT_HOME=~/Desktop/SNR-Project

# Type of the tensorflow installation ['c': CPU, 'nvd': NVIDIA GPU, 'amd': AMD GPU]
TF_VERSION=amd

# Dataset to be downloaded from Kaggle (in form <owner>/<dataset-name>)
DATASET=moltean/fruits

# Name of the docker image (if AMD GPU used)
DOCK_IMG='snr-rocm'

# ----------------------------------------------- Exports --------------------------------------------------------

export PROJECT_HOME=$PROJECT_HOME
export TF_VERSION=$TF_VERSION
export DATASET=$DATASET
export KAGGLE_CONFIG_DIR=$PROJECT_HOME/config/kaggle#
export DOCK_IMG=$DOCK_IMG


# --------------------------------------------- Handy aliases ----------------------------------------------------

if [[ $TF_VERSION == "amd" ]]; then
    source $PROJECT_HOME/scripts/impl/aliases.bash
fi


# --------------------------------------------- Scripts calls ----------------------------------------------------

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
