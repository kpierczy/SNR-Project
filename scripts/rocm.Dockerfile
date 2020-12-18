# ==================================================================================
# According to the users' reports, there is a problem with Polaris GPUs since
# ROCm-3.7. The workaround is to use earlier version of the ROCm at the time.
# Basic container should be switched back to the 'rocm/tensorflow' when the bug 
# will be fixed.
# ==================================================================================

# Pull the official ROCm image
# FROM rocm/tensorflow:rocm3.5-tf2.2-dev
FROM rocm/tensorflow:rocm3.8-tf2.3-dev
# FROM rocm/tensorflow

# Forward environment variables
ARG PROJECT_HOME
ARG TF_VERSION
ARG DATASET
ARG KAGGLE_CONFIG_DIR

ENV PROJECT_HOME ${PROJECT_HOME}
ENV TF_VERSION ${TF_VERSION}
ENV DATASET ${DATASET}
ENV KAGGLE_CONFIG_DIR ${KAGGLE_CONFIG_DIR}

# Set home variable to fit script's requirements
ENV HOME /home

# Create PWD for the project
WORKDIR ${PROJECT_HOME}

# Copy required scripts and config
COPY ./scripts/impl/installs.bash ${HOME}/installs.bash
COPY ./config/env/requirements_amd.py ${HOME}/requirements_amd.py
COPY ./config/env/requirements.py ${HOME}/requirements.py

# Install sudo
RUN /bin/bash -c "apt-get update && apt-get -y install sudo"

# Install required utilities 
RUN /bin/bash -c "source ${HOME}/installs.bash"
# Install python packages
RUN /bin/bash -c "python3 -m pip install -r ${HOME}/requirements.py"
RUN /bin/bash -c "python3 -m pip install -r ${HOME}/requirements_amd.py"

# Cleanup copied files
RUN /bin/bash -c "rm ${HOME}/installs.bash"
RUN /bin/bash -c "rm ${HOME}/requirements.py"
RUN /bin/bash -c "rm ${HOME}/requirements_amd.py"
