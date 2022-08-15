#!/usr/bin/bash

function plot(){
    python scripts/plot_device_data.py "$@"
}

SG_LOC='21f3b020-e571-4d19-981f-42257a197b2f'
THAILAND_LOC='1c7301b2-1edf-11e4-9863-d050992a8771'
# 13.75 / 104.67 (would be in cambodia, but shipped to thailand)
plot --start 2015-2-1-04 --end 2015-2-4-04 --device_id 05D2FF303830594143185511 --fetch_weather

# 1.293 / 103.856 # duplicate of the one below ??
#plot --start 2015-2-22-08 --end 2015-2-26 --device_id 05DDFF303830594143245927 --location_id $SG_LOC

# 1.293 / 103.856
plot --start 2015-2-21-12 --end 2015-2-25 --device_id 05DDFF303830594143185612 --fetch_weather
plot --start 2015-2-25-15 --end 2015-2-25-21 --device_id 05DDFF303830594143185612 --fetch_weather

# 1.293 / 103.856
plot --start 2015-2-22 --end 2015-2-26 --device_id 05D3FF303830594143187512 --fetch_weather

# 22.283 / 114.15 julian heating vs cooling
#plot --start 2015-2-22-08 --end 2015-2-24-08 --device_id 6409FF383939473443067324
#plot --start 2015-2-24-08 --end 2015-2-25-08 --device_id 6409FF383939473443067324

# -37.81 / 114.96 heating vs cooling
plot --start 2015-2-21-12 --end 2015-2-23-12 --device_id 05D9FF303830594143206026 --fetch_weather

# 1.293 / 103.856
plot --start 2015-2-22-08 --end 2015-2-26-08 --device_id 05DDFF303830594143245927 --fetch_weather

# 1.367 / 103.8
plot --start 2015-2-22 --end 2015-2-27 --device_id 05D5FF303830594143207715 --fetch_weather


