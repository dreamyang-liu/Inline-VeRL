docker create --runtime=nvidia --gpus all --net=host --shm-size="32g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl 819eaae2c645 sleep infinity
docker start verl
docker exec -it verl bash