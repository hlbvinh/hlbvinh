#!/bin/bash

./install/systemd/docker_create_containers.sh
sleep 30
./install/systemd/docker_load_data_containers.sh

docker stop mongo redis mysql cassandra
