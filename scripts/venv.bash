# =============================================================================
# Script checks all required dependencies, installs pipenv API (if not present)
# and sets virtual environment for work with Keras.
#
# @note : You should 'source' this script instead of executing.
# =============================================================================

# Project's home directory
PROJECT_HOME=~/Desktop/SNR-Project

# -------------------------------- Dependencies check ------------------------------------------

export PROJECT_HOME=$PROJECT_HOME

# Check if python 3 is available 
if ! which python3 > /dev/null; then
    echo "ERR: No 'python3' available. Cannot run kaggle API. Exiting..."
    exit 1
fi

# Check if virtualenv is available (install, if not)
if ! python3 -m pip list | grep 'virtualenv ' > /dev/null; then
    echo "LOG: 'virtualenv' API will be installed"
    python3 -m pip install --user virtualenv
fi

# ----------------------------------- Set virtualenv ---------------------------------------------

# Create venv folder, if needed 
if [[ ! -d $PROJECT_HOME/config/venv ]]; then
    python3 -m virtualenv $PROJECT_HOME/config/env/venv
fi

# Install required tools
if ! which graphviz > /dev/null; then
    echo "LOG: Installing graphviz dor Keras visualizations"
    sudo apt install graphviz
fi

# Source virtual environment
source $PROJECT_HOME/config/env/venv/bin/activate

# Install required packages in the virtual environment
echo
echo "LOG: Installing required packages in the virtual environment"
echo
pip install --use-feature=2020-resolver -r $PROJECT_HOME/config/env/requirements.txt
