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


# ----------------------------------------------- GPU fan controll -----------------------------------------------

# Get GPU's fan speed in 0-255 range
alias fsp="cat /sys/class/hwmon/hwmon3/pwm1"

# Get GPU's fan mode
alias fspmode="cat /sys/class/hwmon/hwmon3/pwm1_enable"

# Set fans' speed to 'auto' mode
alias fspauto="sudo bash -c 'echo 2 > /sys/class/hwmon/hwmon3/pwm1_enable'"

# Set fans' speed to 'manual' mode
alias fspman="sudo bash -c 'echo 1 > /sys/class/hwmon/hwmon3/pwm1_enable'"

# Set fan speed in 0-255 range (in 'manual' mode)
fspset() { sudo bash -c "echo $1 > /sys/class/hwmon/hwmon3/pwm1"; }
