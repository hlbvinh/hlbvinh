#!/bin/bash
# 
#   usage: ./download_test_data.sh SUFFIX # download schema and data
#          ./download_test_data.sh SUFFIX data # download data
#
#   where SUFFIX is the suffix of a section in the ~/.my.cnf file with valid
#   credentials:
#   
#   [clientSUFFIX]
#   host=db.ambiclimate.com
#   user=...
#   ...
#
CONF=${1:-local}
AMBIDUMP=db/ambinet_test_data.sql
AMBISTRUCTURE=db/ambinet_schema.sql
START_DATE="'2016-7-1'"                    # for sensors (loads of data)
LONG_START_DATE="'2016-6-14'"
END_DATE="'2016-7-3'"  # for sensors (lots of data)
END_DATE_LONG="'2016-7-13'"  # for user geo locations
DEVICES="('EDCFFF303433463443206509')"

rm -f "$AMBIDUMP"

if [[ "$2" != "data" ]]; then
    echo "dumping schema"
    mysqldump --defaults-group-suffix="$CONF" -d --skip-triggers AmbiNet > "$AMBISTRUCTURE"
fi

function ambi_dump(){
    # usage: ambi_dump TableName "device_id IN (...)"
    echo "dumping" "$1"
    mysqldump --defaults-group-suffix="$CONF" --skip-triggers --no-create-info --lock-all-tables --where "$2" AmbiNet "$1" >> "$AMBIDUMP"
}

ambi_dump Device "device_id IN $DEVICES"

APPS="(SELECT appliance_id FROM AmbiNet.DeviceApplianceList WHERE device_id IN $DEVICES)"
APPSTATES="(SELECT appliance_state_id FROM AmbiNet.ApplianceState WHERE appliance_id IN $APPS AND created_on BETWEEN $START_DATE AND $END_DATE)"
ambi_dump DeviceApplianceHistory "appliance_id IN $APPS"
ambi_dump DeviceApplianceList "appliance_id IN $APPS"
ambi_dump ApplianceState "appliance_id IN $APPS AND created_on BETWEEN $LONG_START_DATE AND $END_DATE"
ambi_dump ApplianceControlTarget "device_id IN $DEVICES AND created_on BETWEEN $LONG_START_DATE AND $END_DATE"

LOCATIONS="(SELECT location_id FROM AmbiNet.LocationDeviceList WHERE device_id IN $DEVICES)"
ambi_dump LocationDeviceList "device_id IN $DEVICES"
ambi_dump Location "location_id IN $LOCATIONS"

USERS="(SELECT DISTINCT(user_id) FROM AmbiNet.UserDeviceList WHERE device_id IN $DEVICES)"
ambi_dump User "user_id in $USERS"
ambi_dump UserDeviceList "user_id in $USERS"
ambi_dump UserCheckin "user_id in $USERS AND created_on BETWEEN $START_DATE AND $END_DATE_LONG"

ambi_dump SensorTemperature "device_id IN $DEVICES AND created_on BETWEEN $START_DATE AND $END_DATE"
ambi_dump SensorHumidity "device_id IN $DEVICES AND created_on BETWEEN $START_DATE AND $END_DATE"
ambi_dump SensorPIRCount "device_id IN $DEVICES AND created_on BETWEEN $START_DATE AND $END_DATE"
ambi_dump SensorPIRLoad "device_id IN $DEVICES AND created_on BETWEEN $START_DATE AND $END_DATE"
ambi_dump SensorLuminosity "device_id IN $DEVICES AND created_on BETWEEN $START_DATE AND $END_DATE"

ambi_dump WeatherAPI "location_id in $LOCATIONS AND timestamp BETWEEN $START_DATE AND $END_DATE"

# download same sensor data from cassandra
db/cassandra_sensor_export.py --device_id $DEVICES --start $START_DATE --end $END_DATE --output_filename db/cassandra_sensor_data.csv
# vim: set textwidth=0 wrapmargin=0
