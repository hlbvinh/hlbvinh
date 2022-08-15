#!/bin/bash -e
echo "===================== Installing mysql 5.6 ====================="
echo "===================== Setting mysql defaults ==============================="
mysql -h 127.0.0.1 -uroot -proot -e "CREATE USER 'test'@'%' IDENTIFIED BY 'test'"
mysql -h 127.0.0.1 -uroot -proot -e "CREATE DATABASE Test"
mysql -h 127.0.0.1 -uroot -proot -e "GRANT ALL ON Test.* TO 'test'@'%'"
echo "CREATED TEST DATABASE"
echo "INSTALLING SCHEMA"
mysql -h 127.0.0.1 -uroot -proot -e "CREATE DATABASE IF NOT EXISTS TestAmbiNet"
mysql -h 127.0.0.1 -uroot -proot -e "GRANT ALL ON TestAmbiNet.* TO 'test'@'%'"
mysql -h 127.0.0.1 -uroot -proot --database TestAmbiNet < db/ambinet_schema.sql
echo "SCHEMA INSTALLED"
echo "INSERTING TEST DATA"
mysql -h 127.0.0.1 -uroot -proot --database TestAmbiNet < db/ambinet_test_data.sql
echo "TEST DATA INSERTED"
