# Show running containers
alias dps='sudo docker ps'

# Remove all containers.
alias drm='                                                  \
    if [[ $(sudo docker ps -a -q) != "" ]] > /dev/null; then \
        sudo docker stop $(sudo docker ps -a -q) &&          \
        sudo docker rm $(sudo docker ps -a -q);              \
    fi'

# Remove all containers. Prune intermediate images.
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
