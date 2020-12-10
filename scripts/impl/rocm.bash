# Build the docker image with the preconfigured environment
if ! sudo docker images | grep snr-rocm > /dev/null; then
    echo -e "\nLOG: Building a docker image for ROCm environment.\n"
    alias builder='sudo docker build                                \
        -f scripts/rocm.Dockerfile                                  \
        -t snr-rocm                                                 \
        --build-arg PROJECT_HOME=${PROJECT_HOME}                    \
        --build-arg TF_VERSION=${TF_VERSION}                        \
        --build-arg DATASET=${DATASET}                              \
        --build-arg KAGGLE_CONFIG_DIR={$PROJECT_HOME/config/kaggle}'

    if ! builder .; then
        echo -e "\nERR: Building a docker image failes.\n"
        return
    fi
fi

# Run the container
echo -e "\nLOG: Running virtual environment for ROCm tensorflow \n"
sudo docker run                            \
    -it                                    \
    --rm                                   \
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
    snr-rocm
