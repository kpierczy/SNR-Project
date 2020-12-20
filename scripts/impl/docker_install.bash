# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2020-12-10 17:05:55
# @ Modified time: 2020-12-20 17:50:03
# @ Description:
#
#      Downloads and installs lates stable docker version. If the docker command is found, the script
#      will return instantly.
#
# @note: The script installs docker for amd64 architercture. To change it, modify the docker repository added
#    in the 3rd step.
# ================================================================================================================



# Install docker
if ! which docker > /dev/null; then

    echo -e "\nLOG: Installing docker\n"

    # 1. Install dependancies and download docker's key
    sudo apt-get update
    sudo apt-get install -y  \
            apt-transport-https \
            ca-certificates     \
            curl                \
            gnupg-agent         \
            software-properties-common

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

    # 2. Check docker key's fingerprint
    if ! sudo apt-key fingerprint 0EBFCD88; then
        echo -e "\nERR: Wrong docker's key.\n"
        return 1
    fi
    
    # 3. Set the 'stable' repository for the docker
    sudo add-apt-repository -y \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) \
        stable"

    # 4. Install the docker
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io

fi