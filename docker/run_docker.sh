#!/bin/bash

docker run -it \
    --gpus all \
    --network=host \
    --ipc=host \
    --cap-add=SYS_ADMIN \
    -v /home/zhenlin/workspace/:/home/wuzhenlin/workspace/ \
    -w /home/wuzhenlin/workspace/ \
    --name xgminer \
    xgminer:latest bash

# docker start 
# docker attach
# docker exec
# CTRL-p CTRL-q