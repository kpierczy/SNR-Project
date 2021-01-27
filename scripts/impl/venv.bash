# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2020-12-09 18:16:36
# @ Modified time: 2020-12-20 17:54:57
# @ Description:
#
#     Sets virtual environment for work with Keras (desired to source if CPU or Nvidia GPU tensorflow version
#     used).
#
# @ Note: You should 'source' this script instead of executing.
# ================================================================================================================

# Create venv folder, if needed 
if [[ ! -d /home/erucolindo/Dokumenty/Projekty/Python/SNR-Project/config/venv ]]; then
    python3 -m virtualenv /home/erucolindo/Dokumenty/Projekty/Python/SNR-Project/config/env/venv
fi

# Source virtual environment
source /home/erucolindo/Dokumenty/Projekty/Python/SNR-Project/config/env/venv/bin/activate

# Update pip
python3 -m pip install --upgrade pip

# Install required packages in the virtual environment
echo -e "\nLOG: Installing required packages in the virtual environment\n"
python3 -m pip install -r /home/erucolindo/Dokumenty/Projekty/Python/SNR-Project/config/env/requirements.py

if   [[ $TF_VERSION == "c" ]]; then
	echo cpu version
    python3 -m pip install -r /home/erucolindo/Dokumenty/Projekty/Python/SNR-Project/config/env/requirements_cpu.py
elif [[ $TF_VERSION == "nvd" ]]; then
	echo nvidia version
    python3 -m pip install -r /home/erucolindo/Dokumenty/Projekty/Python/SNR-Project/config/env/requirements_nvd.py
fi
