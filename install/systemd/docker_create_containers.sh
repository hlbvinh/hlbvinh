#!/bin/bash

MONGO_DOCKER_PORT=27017
REDIS_DOCKER_PORT=6379
MYSQL_DOCKER_PORT=3306
CASSANDRA_DOCKER_PORT=9042

docker stop mongo redis mysql cassandra
docker rm mongo redis mysql cassandra

docker run -p $MONGO_DOCKER_PORT:$MONGO_DOCKER_PORT --name mongo -d mongo:3.2
docker run -p $REDIS_DOCKER_PORT:$REDIS_DOCKER_PORT --name redis -d redis:3.0
docker run -p $MYSQL_DOCKER_PORT:$MYSQL_DOCKER_PORT --name mysql -e MYSQL_ROOT_PASSWORD=root -d mysql:5.5
docker run -p $CASSANDRA_DOCKER_PORT:$CASSANDRA_DOCKER_PORT --name cassandra -e JVM_OPTS='-Xmx512m -Xms256m' -d cassandra:3




