# ==============================================================
# Closes all running containers and prune intermediate images
# ==============================================================

if [[ $(sudo docker ps -a -q) != "" ]] > /dev/null; then
    sudo docker stop $(sudo docker ps -a -q)
    sudo docker rm $(sudo docker ps -a -q)
fi
sudo docker image prune