#!/bin/bash

CASSANDRA_DOCKER_PORT=9042

./install/install_mysql.sh
PYTHONPATH=$PWD python db/insert_cassandra_sensor_data.py  --host 127.0.0.1 --port $CASSANDRA_DOCKER_PORT  --fname db/cassandra_sensor_data.csv

docker stop mysql cassandra
