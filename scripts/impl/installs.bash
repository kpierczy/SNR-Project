# =============================================================================
# Script checks all required dependencies, installs:
#    - graphviz (for Keras visualisation)
#    - pipenv API (if not present)
#
# If the TF_VERSION points to amd GPU, installs Docker and a required image.
# 
# @note : You should 'source' this script instead of executing.
# =============================================================================


# -------------------------------- Dependencies check ------------------------------------------


# Check if python 3 is available 
if ! which python3 > /dev/null; then
    echo -e "\nERR: No 'python3' available. Cannot run kaggle API. Exiting...\n"
    return 1
fi


# ----------------------------------- Installations --------------------------------------------

# Update pip
python3 -m pip install --upgrade pip

# Install virtualenv
if ! python3 -m pip list | grep 'virtualenv ' > /dev/null; then
    echo -e "\nLOG: 'virtualenv' API will be installed\n"
    PATH=$PATH:/home/.local/bin
    export PATH=$PATH
    python3 -m pip install --user virtualenv
fi

# Install graphviz
if ! which graphviz > /dev/null; then
    echo -e "\nLOG: Installing graphviz for Keras visualizations\n"
    sudo apt install -y graphviz
fi
