DROP TABLE IF EXISTS Research.WeatherAPI;
CREATE TABLE Research.WeatherAPI (
    `row_id` INT(11) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    `latitude` DECIMAL(7, 5) NOT NULL,
    `longitude` DECIMAL(8, 5) NOT NULL,
    `timestamp` DATETIME NOT NULL,
    `precip_intensity` FLOAT,
    `precip_probability` FLOAT,
    `temperature` FLOAT,
    `apparent_temperature` FLOAT,
    `dew_point` FLOAT,
    `humidity` FLOAT,
    `wind_speed` FLOAT,
    `wind_bearing` FLOAT,
    `visibility` FLOAT,
    `cloud_cover` FLOAT,
    `pressure` FLOAT,
    `ozone` FLOAT
);
