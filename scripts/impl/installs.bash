# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2020-12-09 18:20:23
# @ Modified time: 2020-12-20 18:15:56
# @ Description:
#
#     Script checks required dependencies and installs it when lacking:
#     - graphviz (for Keras visualisation)
#     - pipenv API (if not present)
# 
# @ Note : You should 'source' this script instead of executing as it adds some paths into the 'PATH'
#     environment variable.
# ================================================================================================================


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
if ! apt list --installed 2> /dev/null | awk '/graphviz/ {print $1}' > /dev/null; then
    echo -e "\nLOG: Installing graphviz for Keras visualizations\n"
    sudo apt install -y graphviz
fi
