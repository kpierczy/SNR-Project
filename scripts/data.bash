# =============================================================================
# Script checks all required dependencies, installs kaggle API (if not present)
# and downloads choosen dataset.
#
# @note : Read 'config/kaggle/README.md' first
# =============================================================================

# Project's home directory
PROJECT_HOME=~/Desktop/SNR-Project

# Dataset to be downloaded (in form <owner>/<dataset-name>)
DATASET=moltean/fruits

# -------------------------------- Dependencies check ------------------------------------------

export KAGGLE_CONFIG_DIR=$PROJECT_HOME/config/kaggle

# Check if python 3 is available 
if ! which python3 > /dev/null; then
    echo "ERR: No 'python3' available. Cannot run kaggle API. Exiting..."
    exit 1
fi

# Check if kaggle API is installed (install, if not)
if ! which kaggle > /dev/null; then
    echo "LOG: kaggle API will be installed"
    python3 -m pip install --user kaggle
fi

# --------------------------------- Dataset download -------------------------------------------

# Download dataset
if [[ ! -d $PROJECT_HOME/data/fruits-360 && ! -f $PROJECT_HOME/data/fruits.zip ]]; then
    echo "LOG: Downloading $DATASET dataset"
    kaggle datasets download -p$PROJECT_HOME/data $DATASET
fi

# Unzip downloaded data
if [[ ! -d $PROJECT_HOME/data/fruits-360 ]]; then
    
    # Check if unzip is available
    if ! which unzip > /dev/null; then
        echo "ERR: 'unzip' tool unavailable. Exiting..."
        exit 1
    fi

    echo "LOG: Unzipping dataset"
    unzip $PROJECT_HOME/data/fruits.zip -d $PROJECT_HOME/data > /dev/null
fi

# Delete zip file
if [[ -f $PROJECT_HOME/data/fruits.zip ]]; then
    echo "LOG: Removing data/fruits.zip"
    rm $PROJECT_HOME/data/fruits.zip
fi