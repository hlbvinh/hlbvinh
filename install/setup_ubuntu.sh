#!/bin/bash
set -e

# mongodb
apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv EA312927
echo "deb http://repo.mongodb.org/apt/ubuntu trusty/mongodb-org/3.2 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-3.2.list

add-apt-repository -y ppa:deadsnakes/ppa
add-apt-repository -y ppa:chris-lea/redis-server
apt-get update
apt-get upgrade -y

apt-get -y install build-essential \
        curl \
        git \
        libssl-dev \
        libffi-dev \
        mongodb-org \
        mysql-client \
        python3.6-dev \
        redis-server
