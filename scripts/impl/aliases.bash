# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2020-12-10 23:06:00
# @ Modified time: 2020-12-20 17:42:34
# @ Description:
#
#     Sourcing this script will provide the user's terminal with a set of handy commands used widely
#     during working on the project. The main reason to use these commands is to reduce number of
#     click's performed by the user when interacting with AMD-GPU based system :*
#
# ================================================================================================================

# ----------------------------------------------- Docker utilities -----------------------------------------------

if which docker > /dev/null; then

    # Show running containers
    alias dps='sudo docker ps'

    # Stop and remove all containers
    alias drm='                                                  \
        if [[ $(sudo docker ps -a -q) != "" ]] > /dev/null; then \
            sudo docker stop $(sudo docker ps -a -q) &&          \
            sudo docker rm $(sudo docker ps -a -q);              \
        fi'

    # Stop and remove all containers. Prune intermediate images.
    alias prune='                                                \
        if [[ $(sudo docker ps -a -q) != "" ]] > /dev/null; then \
            sudo docker stop $(sudo docker ps -a -q) &&          \
            sudo docker rm $(sudo docker ps -a -q);              \
        fi && sudo docker image prune'

    # Show docker images
    alias dimg='sudo docker images'

    # Remove a docker image
    alias dimgrm='sudo docker rmi'

    # Executes an additional bash in the running environment
    if [[ $DOCK_IMG != "" ]]; then
        alias dexec="sudo docker exec -it $(dps | awk -v i=$DOCK_IMG '/i/ {print $1}') bash"
    else
        alias dexec="sudo docker exec -it $(dps | awk '/snr-rocm/ {print $1}') bash"
    fi
fi


# --------------------------------------------- Neural nets workflow ---------------------------------------------

# Clear all models logs and history files from the given mode's directory
nncl() {
    sudo rm -rf models/$1/logs/*
    sudo rm -rf models/$1/history/*
    sudo rm -rf models/$1/test/*
    sudo rm -rf models/$1/*.hdf5
}

# Opens tensorboard with data of the given model
tboard() {
    tensorboard --logdir_spec training:models/$1/logs,test:models/$1/test
}
