# =============================================================================
# Script downloads choosen dataset.
#
# @note : Read 'config/kaggle/README.md' first
# =============================================================================

# Check if python 3 is available 
if ! which python3 > /dev/null; then
    echo -e "\nERR: No 'python3' available. Cannot run kaggle API. Exiting...\n"
    return 1
fi

# -------------------------------- Dependencies check ------------------------------------------

# Install Kaggle API
if ! which kaggle > /dev/null; then
    echo -e "\nLOG: kaggle API will be installed\n"
    python3 -m pip install --user kaggle
fi

# --------------------------------- Dataset download -------------------------------------------

# Download dataset
if [[ ! -d $PROJECT_HOME/data/fruits-360 && ! -f $PROJECT_HOME/data/fruits.zip ]]; then
    echo -e "\nLOG: Downloading $DATASET dataset\n"
    kaggle datasets download -p $PROJECT_HOME/data $DATASET
fi

# Unzip downloaded data
if [[ ! -d $PROJECT_HOME/data/fruits-360 ]]; then
    
    # Check if unzip is available
    if ! which unzip > /dev/null; then
        echo -e "\nERR: 'unzip' tool unavailable. Exiting...\n"
        exit 1
    fi

    echo -e "\nLOG: Unzipping dataset\n"
    unzip $PROJECT_HOME/data/fruits.zip -d $PROJECT_HOME/data > /dev/null
fi

# Delete zip file
if [[ -f $PROJECT_HOME/data/fruits.zip ]]; then
    echo -e "\nLOG: Removing data/fruits.zip\n"
    rm $PROJECT_HOME/data/fruits.zip
fi