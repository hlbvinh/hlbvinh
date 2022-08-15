#!/bin/bash

cp install/systemd/*.service /etc/systemd/system/
systemctl daemon-reload
systemctl start {mysql,cassandra,redis,mongodb}-docker
systemctl enable {mysql,cassandra,redis,mongodb}-docker

