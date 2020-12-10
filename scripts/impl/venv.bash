# =============================================================================
# Script sets virtual environment for work with Keras (desired to source
# if CPU or Nvidia GPU tensorflow version used).
#
# @note : You should 'source' this script instead of executing.
# =============================================================================

# ----------------------------------- Set virtualenv ---------------------------------------------

# Create venv folder, if needed 
if [[ ! -d $PROJECT_HOME/config/venv ]]; then
    python3 -m virtualenv $PROJECT_HOME/config/env/venv
fi

# Source virtual environment
source $PROJECT_HOME/config/env/venv/bin/activate

# Install required packages in the virtual environment
echo -e "\nLOG: Installing required packages in the virtual environment\n"
python3 -m pip install -r $PROJECT_HOME/config/env/requirements.py

if   [[ $TF_VERSION == "c" ]]; then
    python3 -m pip install -r $PROJECT_HOME/config/env/requirements_cpu.py
elif [[ $TF_VERSION == "nvd" ]]; then
    python3 -m pip install -r $PROJECT_HOME/config/env/requirements_nvd.py
fi
