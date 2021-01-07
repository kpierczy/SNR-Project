# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2020-12-10 15:53:14
# @ Modified time: 2020-12-20 17:52:58
# @ Description:
#
#      Builds and runs the docker image used to work with AMD-GPU-base systems.
# ================================================================================================================

# Build the docker image with the preconfigured environment
if ! sudo docker images | grep $DOCK_IMG > /dev/null; then
    printf "\nLOG: Building a docker image for ROCm environment.\n"
    builder="sudo docker build                                      \
        -f $PROJECT_HOME/scripts/rocm.Dockerfile                    \
        -t $DOCK_IMG                                                \
        --build-arg PROJECT_HOME=${PROJECT_HOME}                    \
        --build-arg TF_VERSION=${TF_VERSION}                        \
        --build-arg DATASET=${DATASET}                              \
        --build-arg KAGGLE_CONFIG_DIR={$PROJECT_HOME/config/kaggle}"

    if ! $builder .; then
        printf "\nERR: Building a docker image failes.\n"
        return
    fi
fi

# Run the container
printf "\nLOG: Running virtual environment for ROCm tensorflow \n\n"
sudo docker run                            \
    -it                                    \
    --rm                                   \
    --name $DOCK_IMG                       \
    --network=host                         \
    --device=/dev/kfd                      \
    --device=/dev/dri                      \
    --ipc=host                             \
    --shm-size 16G                         \
    --group-add video                      \
    --cap-add=SYS_PTRACE                   \
    --security-opt seccomp=unconfined      \
    -v $HOME/dockerx:/dockerx              \
    -v $PROJECT_HOME:$PROJECT_HOME         \
    $DOCK_IMG    
