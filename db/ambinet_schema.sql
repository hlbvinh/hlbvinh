-- MySQL dump 10.13  Distrib 5.5.50, for Linux (x86_64)
--
-- Host: db-slave    Database: AmbiNet
-- ------------------------------------------------------
-- Server version	5.5.46-0ubuntu0.14.04.2-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ACEventHistory`
--

DROP TABLE IF EXISTS `ACEventHistory`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ACEventHistory` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `trigger_id` int(11) NOT NULL,
  `fire_at` datetime NOT NULL,
  `status` enum('OK','FAILED','MISSED') COLLATE utf8_unicode_ci NOT NULL DEFAULT 'OK',
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`id`),
  KEY `trigger_id` (`trigger_id`)
) ENGINE=InnoDB AUTO_INCREMENT=58259 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ACEventTrigger`
--

DROP TABLE IF EXISTS `ACEventTrigger`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ACEventTrigger` (
  `trigger_id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `device_id` varchar(24) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `name` varchar(24) COLLATE utf8_unicode_ci NOT NULL DEFAULT 'Trigger',
  `trigger_type` varchar(16) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `action` text COLLATE utf8_unicode_ci NOT NULL,
  `created_on` datetime NOT NULL,
  `trigger_rule` text COLLATE utf8_unicode_ci NOT NULL,
  `enabled` tinyint(1) NOT NULL DEFAULT '1',
  PRIMARY KEY (`trigger_id`),
  KEY `event_user_id_fk` (`user_id`),
  KEY `event_device_id_fk` (`device_id`),
  CONSTRAINT `event_device_id_fk` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`) ON DELETE CASCADE,
  CONSTRAINT `event_user_id_fk` FOREIGN KEY (`user_id`) REFERENCES `User` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=1456 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ACTimer`
--

DROP TABLE IF EXISTS `ACTimer`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ACTimer` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `trigger_id` int(11) unsigned zerofill NOT NULL,
  `next_execution` datetime DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `trigger_id` (`trigger_id`),
  KEY `next_execution` (`next_execution`),
  CONSTRAINT `ACTimer_ibfk_1` FOREIGN KEY (`trigger_id`) REFERENCES `ACEventTrigger` (`trigger_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=1434 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `AccessToken`
--

DROP TABLE IF EXISTS `AccessToken`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `AccessToken` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `token_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `ttype` set('Bearer') COLLATE utf8_unicode_ci NOT NULL,
  `created_on` datetime NOT NULL,
  `expires_in` int(10) unsigned NOT NULL,
  `last_activity` datetime DEFAULT NULL,
  `scope` set('REST','Mobile','Web','Admin') COLLATE utf8_unicode_ci NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `user_id` (`user_id`),
  KEY `token_id` (`token_id`),
  CONSTRAINT `AccessToken_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `User` (`user_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=297236 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ApplianceControlTarget`
--

DROP TABLE IF EXISTS `ApplianceControlTarget`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ApplianceControlTarget` (
  `row_id` int(11) NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `quantity` enum('Ambi','Away_Humidity_Upper','Humidex','Humidity','Temperature','Manual','Away_Temperature_Upper','Away_Humidex_Upper','Off','Away_Temperature_Lower','Away_Humidex_Lower','Away_Humidity_Lower','Climate') COLLATE utf8_unicode_ci DEFAULT NULL,
  `value` float DEFAULT NULL,
  `signal_interval` time NOT NULL DEFAULT '00:10:00',
  `created_on` datetime NOT NULL,
  `origin` enum('User','Timer','Reverse','Geo') COLLATE utf8_unicode_ci DEFAULT NULL,
  PRIMARY KEY (`row_id`),
  KEY `device_id` (`device_id`),
  KEY `device_id_2` (`device_id`,`created_on`),
  KEY `device_id_row_id` (`device_id`,`row_id`),
  CONSTRAINT `ApplianceControlTarget_ibfk_1` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`)
) ENGINE=InnoDB AUTO_INCREMENT=801801 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ApplianceProfile`
--

DROP TABLE IF EXISTS `ApplianceProfile`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ApplianceProfile` (
  `appliance_profile_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `ircode_profile_id` int(10) unsigned DEFAULT NULL,
  `irpn_id` int(10) unsigned DEFAULT NULL,
  `manufacturer` varchar(50) COLLATE utf8_unicode_ci NOT NULL,
  `model` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `series` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `name` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `type` enum('AirConditioner','Heater','Light','Fan','Blinds') COLLATE utf8_unicode_ci NOT NULL,
  PRIMARY KEY (`appliance_profile_id`),
  KEY `ircode_profile_id` (`ircode_profile_id`),
  KEY `irpn_id` (`irpn_id`),
  CONSTRAINT `ApplianceProfile_ibfk_1` FOREIGN KEY (`ircode_profile_id`) REFERENCES `IRCodeProfile` (`ircode_profile_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ApplianceState`
--

DROP TABLE IF EXISTS `ApplianceState`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ApplianceState` (
  `appliance_state_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `appliance_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `power` enum('On','Off') COLLATE utf8_unicode_ci DEFAULT NULL,
  `mode` enum('Auto','Cool','Dry','Fan','Heat') COLLATE utf8_unicode_ci DEFAULT NULL,
  `fan` enum('Auto','High','Med','Low','Med-Low','Med-High','Very-High','Quiet','Night','Off') COLLATE utf8_unicode_ci DEFAULT NULL,
  `temperature` int(11) DEFAULT NULL,
  `swing` enum('On','Off','Auto','Left','Mid-Left','Mid','Mid-Right','Right','Oscillate','Left-Right','Up-Down','Both') COLLATE utf8_unicode_ci DEFAULT NULL,
  `louver` enum('Up','Mid','Down','On','Off','Mid-Up','Mid-Down','Auto','Oscillate','Swing','Up-High','Up-Low','Mid-High','Mid-Low','Down-High','Down-Low') COLLATE utf8_unicode_ci DEFAULT NULL,
  `button` varchar(24) COLLATE utf8_unicode_ci DEFAULT NULL,
  `origin` varchar(12) COLLATE utf8_unicode_ci DEFAULT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`appliance_state_id`),
  KEY `appliance_id` (`appliance_id`),
  KEY `appliance_id_2` (`appliance_id`,`created_on`),
  KEY `created_on` (`created_on`),
  KEY `appliance_id_and_appliance_state_id` (`appliance_id`,`appliance_state_id`) COMMENT 'speedup dashboard query(cant use created_on)',
  KEY `appliance_id_power` (`appliance_id`,`power`) COMMENT 'query latest power on states',
  KEY `appliance_power_state` (`appliance_id`,`power`,`appliance_state_id`) COMMENT 'for latest power On states',
  CONSTRAINT `ApplianceState_ibfk_1` FOREIGN KEY (`appliance_id`) REFERENCES `DeviceApplianceList` (`appliance_id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB AUTO_INCREMENT=1950533 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `DeploymentHistory`
--

DROP TABLE IF EXISTS `DeploymentHistory`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `DeploymentHistory` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `source` enum('irdeployment','skynet','unknown','test','testremote','skynet_timer','skynet_geo','OpenAPI') COLLATE utf8_unicode_ci NOT NULL DEFAULT 'unknown',
  `source_id` varchar(36) COLLATE utf8_unicode_ci DEFAULT NULL,
  `status` enum('OK','IRP Failure','Device Offline','Unknown') COLLATE utf8_unicode_ci NOT NULL DEFAULT 'Unknown',
  `appliance_state_id` int(10) unsigned DEFAULT NULL,
  `created_on` datetime DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `device_id_idx` (`device_id`,`created_on`),
  KEY `appliance_state_id_idx` (`appliance_state_id`)
) ENGINE=InnoDB AUTO_INCREMENT=1233527 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Device`
--

DROP TABLE IF EXISTS `Device`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Device` (
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `serial_num` varchar(12) COLLATE utf8_unicode_ci DEFAULT NULL,
  `created_on` datetime NOT NULL,
  `type` enum('IRBlaster','PowerSocket','IRAudioJack','') COLLATE utf8_unicode_ci NOT NULL,
  `name` varchar(36) COLLATE utf8_unicode_ci DEFAULT NULL,
  `room_name` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `firmware` varchar(36) COLLATE utf8_unicode_ci DEFAULT NULL,
  `software` varchar(36) COLLATE utf8_unicode_ci DEFAULT NULL,
  `hardware` varchar(36) COLLATE utf8_unicode_ci DEFAULT NULL,
  `manufacturer` varchar(36) COLLATE utf8_unicode_ci DEFAULT NULL,
  `mac` varchar(17) COLLATE utf8_unicode_ci DEFAULT NULL,
  `memory_shared` int(11) DEFAULT NULL,
  `memory_max` int(11) DEFAULT NULL,
  `start_condition` varchar(64) COLLATE utf8_unicode_ci NOT NULL DEFAULT '10000 10000',
  `timeout` int(10) unsigned NOT NULL DEFAULT '65536' COMMENT 'DEPRECATED',
  `buzzer` tinyint(1) NOT NULL DEFAULT '1' COMMENT 'DEPRECATED',
  `irlog` tinyint(1) NOT NULL DEFAULT '0',
  `ir_recv` enum('0','33','38','56') COLLATE utf8_unicode_ci NOT NULL DEFAULT '38',
  `timezone` varchar(32) COLLATE utf8_unicode_ci DEFAULT NULL,
  PRIMARY KEY (`device_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `DeviceApplianceHistory`
--

DROP TABLE IF EXISTS `DeviceApplianceHistory`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `DeviceApplianceHistory` (
  `device_appliance_history_id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `appliance_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `irprofile_id` int(10) unsigned DEFAULT NULL,
  `start` datetime NOT NULL,
  `end` datetime DEFAULT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`device_appliance_history_id`),
  KEY `device_id` (`device_id`,`appliance_id`),
  KEY `appliance_id` (`appliance_id`),
  KEY `start` (`start`),
  KEY `end` (`end`),
  CONSTRAINT `DeviceApplianceHistory_ibfk_1` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`) ON DELETE NO ACTION,
  CONSTRAINT `DeviceApplianceHistory_ibfk_3` FOREIGN KEY (`appliance_id`) REFERENCES `DeviceApplianceList` (`appliance_id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB AUTO_INCREMENT=3467 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `DeviceApplianceList`
--

DROP TABLE IF EXISTS `DeviceApplianceList`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `DeviceApplianceList` (
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `appliance_profile_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `appliance_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `irprofile_id` int(10) unsigned DEFAULT NULL,
  PRIMARY KEY (`appliance_id`),
  KEY `device_id` (`device_id`),
  KEY `appliance_profile_id` (`appliance_profile_id`),
  KEY `irprofile_id` (`irprofile_id`),
  CONSTRAINT `DeviceApplianceList_ibfk_1` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`),
  CONSTRAINT `DeviceApplianceList_ibfk_2` FOREIGN KEY (`appliance_profile_id`) REFERENCES `ApplianceProfile` (`appliance_profile_id`),
  CONSTRAINT `DeviceApplianceList_ibfk_3` FOREIGN KEY (`irprofile_id`) REFERENCES `IRProfile` (`irprofile_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `DeviceConnection`
--

DROP TABLE IF EXISTS `DeviceConnection`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `DeviceConnection` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `connection` tinyint(1) NOT NULL,
  `ip` varchar(15) COLLATE utf8_unicode_ci DEFAULT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`row_id`)
) ENGINE=InnoDB AUTO_INCREMENT=1464248 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `DeviceGroup`
--

DROP TABLE IF EXISTS `DeviceGroup`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `DeviceGroup` (
  `row_id` int(10) unsigned NOT NULL DEFAULT '0',
  `group` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  UNIQUE KEY `group` (`group`,`device_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `DeviceSensorList`
--

DROP TABLE IF EXISTS `DeviceSensorList`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `DeviceSensorList` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `sensor_type` enum('Temperature','Humidity','Light','PIR','Sound') COLLATE utf8_unicode_ci NOT NULL,
  PRIMARY KEY (`row_id`),
  UNIQUE KEY `device_id_2` (`device_id`,`sensor_type`),
  KEY `device_id` (`device_id`),
  CONSTRAINT `DeviceSensorList_ibfk_1` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`)
) ENGINE=InnoDB AUTO_INCREMENT=8533 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `DeviceSettings`
--

DROP TABLE IF EXISTS `DeviceSettings`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `DeviceSettings` (
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `auto_led_brightness` tinyint(1) NOT NULL DEFAULT '1',
  `led_brightness` smallint(8) unsigned NOT NULL DEFAULT '5000',
  `buzzer_toggle` tinyint(1) NOT NULL DEFAULT '1',
  `buzzer_volume` tinyint(11) unsigned NOT NULL DEFAULT '24',
  `start_condition` varchar(11) COLLATE utf8_unicode_ci NOT NULL DEFAULT '10000 10000',
  `timeout` mediumint(11) unsigned NOT NULL DEFAULT '65535',
  `tx_power` tinyint(3) unsigned NOT NULL DEFAULT '30',
  PRIMARY KEY (`device_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `FailedReverse`
--

DROP TABLE IF EXISTS `FailedReverse`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `FailedReverse` (
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `irprofile_id` int(10) DEFAULT NULL,
  `capture` varchar(2048) COLLATE utf8_unicode_ci NOT NULL,
  `created_on` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IRChecksum`
--

DROP TABLE IF EXISTS `IRChecksum`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `IRChecksum` (
  `irchecksum_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `ctype` enum('eval','func') COLLATE utf8_unicode_ci NOT NULL,
  `value` varchar(512) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  PRIMARY KEY (`irchecksum_id`)
) ENGINE=InnoDB AUTO_INCREMENT=823 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IRCodeProfile`
--

DROP TABLE IF EXISTS `IRCodeProfile`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `IRCodeProfile` (
  `ircode_profile_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `source` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `source_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `type` enum('Raw','Binary') COLLATE utf8_unicode_ci NOT NULL,
  `modular_frequency` int(11) NOT NULL,
  `max_transitions` int(10) unsigned NOT NULL,
  `shortest_pulse` int(10) unsigned NOT NULL,
  `longest_pulse` int(10) unsigned NOT NULL,
  `shortest_transition` int(10) unsigned NOT NULL,
  `longest_transition` int(10) unsigned NOT NULL,
  `longest_length` int(10) unsigned NOT NULL,
  PRIMARY KEY (`ircode_profile_id`)
) ENGINE=InnoDB AUTO_INCREMENT=3598 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IRCodeRaw`
--

DROP TABLE IF EXISTS `IRCodeRaw`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `IRCodeRaw` (
  `ircode_raw_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `ircode_profile_id` int(10) unsigned NOT NULL,
  `key` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `payload` text COLLATE utf8_unicode_ci NOT NULL,
  `power` enum('Off','On','','') COLLATE utf8_unicode_ci DEFAULT NULL,
  `mode` enum('Auto','Cool','Dry','Fan','Heat') COLLATE utf8_unicode_ci DEFAULT NULL,
  `fan` enum('Auto','High','Mid','Low') COLLATE utf8_unicode_ci DEFAULT NULL,
  `temperature` int(10) unsigned DEFAULT NULL,
  `louver` enum('Up','Mid','Down','On','Off') COLLATE utf8_unicode_ci DEFAULT NULL,
  `swing` enum('Off','On','','') COLLATE utf8_unicode_ci DEFAULT NULL,
  `button` enum('Power','TempDown','TempUp','Swing','Louver','Mode','Fan') COLLATE utf8_unicode_ci DEFAULT NULL,
  PRIMARY KEY (`ircode_raw_id`),
  KEY `key` (`key`),
  KEY `ircode_profile_id` (`ircode_profile_id`),
  CONSTRAINT `IRCodeRaw_ibfk_1` FOREIGN KEY (`ircode_profile_id`) REFERENCES `IRCodeProfile` (`ircode_profile_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IRCondition`
--

DROP TABLE IF EXISTS `IRCondition`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `IRCondition` (
  `ircondition_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `group` int(10) unsigned NOT NULL,
  `irstructure_id` int(10) unsigned NOT NULL,
  `a` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `operator` enum('equals','not_equals','less_than','greater_than') COLLATE utf8_unicode_ci NOT NULL,
  `b` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  PRIMARY KEY (`ircondition_id`),
  KEY `rule_id` (`irstructure_id`),
  KEY `irstructure_id` (`irstructure_id`),
  CONSTRAINT `IRCondition_ibfk_1` FOREIGN KEY (`irstructure_id`) REFERENCES `IRStructure` (`irstructure_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2976 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IRDefinition`
--

DROP TABLE IF EXISTS `IRDefinition`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `IRDefinition` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `irprofile_id` int(10) unsigned NOT NULL,
  `dtype` set('power','mode','fan','temperature','swing','louver','checksum','button','timer') COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `label` varchar(16) COLLATE utf8_unicode_ci DEFAULT NULL,
  `value` int(11) unsigned NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `profile_id` (`irprofile_id`),
  CONSTRAINT `IRDefinition_ibfk_1` FOREIGN KEY (`irprofile_id`) REFERENCES `IRProfile` (`irprofile_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=29497 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IRFeature2`
--

DROP TABLE IF EXISTS `IRFeature2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `IRFeature2` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `irprofile_id` int(10) unsigned NOT NULL,
  `group_name` enum('auto','cool','fan','dry','heat') COLLATE utf8_unicode_ci NOT NULL,
  `feature_name` enum('fan','temperature','swing','louver','swing_inc','louver_inc') COLLATE utf8_unicode_ci NOT NULL,
  `ftype` enum('checkbox','radio','button','select_option') COLLATE utf8_unicode_ci NOT NULL,
  `value` varchar(16) COLLATE utf8_unicode_ci DEFAULT NULL,
  PRIMARY KEY (`row_id`),
  KEY `irprofile_id` (`irprofile_id`),
  CONSTRAINT `IRFeature2_ibfk_1` FOREIGN KEY (`irprofile_id`) REFERENCES `IRProfile` (`irprofile_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=71402 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IRProfile`
--

DROP TABLE IF EXISTS `IRProfile`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `IRProfile` (
  `irprofile_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(64) COLLATE utf8_unicode_ci NOT NULL,
  `model` varchar(64) COLLATE utf8_unicode_ci NOT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`irprofile_id`)
) ENGINE=InnoDB AUTO_INCREMENT=944 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IRReverse`
--

DROP TABLE IF EXISTS `IRReverse`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `IRReverse` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `irprofile_id` int(10) unsigned NOT NULL,
  `irstructure_id` int(10) unsigned NOT NULL,
  `key` varchar(16) COLLATE utf8_unicode_ci DEFAULT NULL,
  `value` varchar(16) COLLATE utf8_unicode_ci DEFAULT NULL,
  `msg_part` int(10) unsigned NOT NULL DEFAULT '0',
  PRIMARY KEY (`row_id`),
  KEY `irprofile_id` (`irprofile_id`),
  CONSTRAINT `IRReverse_ibfk_1` FOREIGN KEY (`irprofile_id`) REFERENCES `IRProfile` (`irprofile_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2662 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IRStructure`
--

DROP TABLE IF EXISTS `IRStructure`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `IRStructure` (
  `irstructure_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `irprofile_id` int(10) unsigned NOT NULL,
  `property` set('power','mode','fan','temperature','swing','louver','checksum','button') COLLATE utf8_unicode_ci DEFAULT NULL,
  `irp` varchar(512) COLLATE utf8_unicode_ci NOT NULL,
  `irchecksum_id` int(10) unsigned DEFAULT NULL,
  `is_default` tinyint(1) NOT NULL DEFAULT '0',
  PRIMARY KEY (`irstructure_id`),
  KEY `profile_id` (`irprofile_id`),
  CONSTRAINT `IRStructure_ibfk_1` FOREIGN KEY (`irprofile_id`) REFERENCES `IRProfile` (`irprofile_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2704 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Location`
--

DROP TABLE IF EXISTS `Location`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Location` (
  `location_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `latitude` double NOT NULL,
  `longitude` double NOT NULL,
  `name` varchar(50) COLLATE utf8_unicode_ci NOT NULL,
  `altitude` double DEFAULT NULL,
  `accuracy` double DEFAULT NULL,
  `tolerance_horizontal` double DEFAULT NULL,
  `tolerance_vertical` double DEFAULT NULL,
  `street1` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `street2` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `street3` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `town_city` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `state_province` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `country` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `zip` varchar(10) COLLATE utf8_unicode_ci DEFAULT NULL,
  `timezone` varchar(32) COLLATE utf8_unicode_ci DEFAULT NULL,
  PRIMARY KEY (`location_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `LocationDeviceHistory`
--

DROP TABLE IF EXISTS `LocationDeviceHistory`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `LocationDeviceHistory` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `location_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `start` datetime NOT NULL,
  `end` datetime DEFAULT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `device_id` (`device_id`,`location_id`),
  KEY `appliance_id` (`location_id`),
  CONSTRAINT `LocationDeviceHistory_ibfk_1` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`),
  CONSTRAINT `LocationDeviceHistory_ibfk_2` FOREIGN KEY (`location_id`) REFERENCES `Location` (`location_id`)
) ENGINE=InnoDB AUTO_INCREMENT=3397 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `LocationDeviceList`
--

DROP TABLE IF EXISTS `LocationDeviceList`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `LocationDeviceList` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `location_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `location_id` (`location_id`),
  KEY `device_id` (`device_id`),
  CONSTRAINT `LocationDeviceList_ibfk_1` FOREIGN KEY (`location_id`) REFERENCES `Location` (`location_id`),
  CONSTRAINT `LocationDeviceList_ibfk_2` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`)
) ENGINE=InnoDB AUTO_INCREMENT=3364 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `PasswordReset`
--

DROP TABLE IF EXISTS `PasswordReset`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `PasswordReset` (
  `user_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `reset_token` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `created_on` datetime NOT NULL,
  UNIQUE KEY `uniqtoken` (`reset_token`),
  KEY `user_id_idx` (`user_id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `pwreset_userid_fk` FOREIGN KEY (`user_id`) REFERENCES `User` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `PendingUser`
--

DROP TABLE IF EXISTS `PendingUser`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `PendingUser` (
  `email` varchar(64) COLLATE utf8_unicode_ci NOT NULL,
  `created_on` datetime NOT NULL,
  `token` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `credential` varchar(64) COLLATE utf8_unicode_ci NOT NULL,
  `from` enum('kickstarter','echelon','ambi','') COLLATE utf8_unicode_ci DEFAULT NULL,
  `ks_account` varchar(32) COLLATE utf8_unicode_ci DEFAULT NULL,
  `first_name` varchar(32) COLLATE utf8_unicode_ci DEFAULT NULL,
  `last_name` varchar(32) COLLATE utf8_unicode_ci DEFAULT NULL,
  PRIMARY KEY (`token`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Remote`
--

DROP TABLE IF EXISTS `Remote`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Remote` (
  `remote_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `manufacture` varchar(64) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `model` varchar(64) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `is_universal` tinyint(1) NOT NULL DEFAULT '0',
  `remote_compatible_id` int(10) unsigned DEFAULT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`remote_id`)
) ENGINE=InnoDB AUTO_INCREMENT=812 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `RemoteCompatible`
--

DROP TABLE IF EXISTS `RemoteCompatible`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `RemoteCompatible` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `remote_compatible_id` int(10) unsigned NOT NULL,
  `manufacture` varchar(64) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  PRIMARY KEY (`row_id`)
) ENGINE=InnoDB AUTO_INCREMENT=20 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `RemoteIRProfileList`
--

DROP TABLE IF EXISTS `RemoteIRProfileList`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `RemoteIRProfileList` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `remote_id` int(10) unsigned NOT NULL,
  `irprofile_id` int(10) unsigned NOT NULL,
  PRIMARY KEY (`id`),
  KEY `remote_id` (`remote_id`),
  KEY `irprofile_id` (`irprofile_id`),
  CONSTRAINT `RemoteIRProfileList_ibfk_1` FOREIGN KEY (`remote_id`) REFERENCES `Remote` (`remote_id`),
  CONSTRAINT `RemoteIRProfileList_ibfk_2` FOREIGN KEY (`irprofile_id`) REFERENCES `IRProfile` (`irprofile_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `SensorAirFlow`
--

DROP TABLE IF EXISTS `SensorAirFlow`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `SensorAirFlow` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `value` varchar(16) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `device_id_2` (`device_id`,`created_on`),
  CONSTRAINT `SensorAirFlow_ibfk_1` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`)
) ENGINE=InnoDB AUTO_INCREMENT=1331987 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `SensorHumidity`
--

DROP TABLE IF EXISTS `SensorHumidity`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `SensorHumidity` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `value` float NOT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `device_id_2` (`device_id`,`created_on`),
  CONSTRAINT `_SensorHumidity_ibfk_1` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`)
) ENGINE=InnoDB AUTO_INCREMENT=507176838 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `SensorInfrared`
--

DROP TABLE IF EXISTS `SensorInfrared`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `SensorInfrared` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `value` text COLLATE utf8_unicode_ci NOT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `device_id` (`device_id`,`created_on`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `SensorLuminosity`
--

DROP TABLE IF EXISTS `SensorLuminosity`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `SensorLuminosity` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `overall_lux` int(10) unsigned NOT NULL,
  `full_spectrum` int(10) unsigned NOT NULL,
  `infrared_spectrum` int(10) unsigned NOT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `device_id_2` (`device_id`,`created_on`)
) ENGINE=InnoDB AUTO_INCREMENT=501425956 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `SensorPIRCount`
--

DROP TABLE IF EXISTS `SensorPIRCount`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `SensorPIRCount` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `value` int(10) unsigned NOT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `device_id_2` (`device_id`,`created_on`)
) ENGINE=InnoDB AUTO_INCREMENT=492744544 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `SensorPIRLoad`
--

DROP TABLE IF EXISTS `SensorPIRLoad`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `SensorPIRLoad` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `one_min` int(10) unsigned NOT NULL,
  `five_min` int(10) unsigned NOT NULL,
  `fifteen_min` int(10) unsigned NOT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `device_id_2` (`device_id`,`created_on`)
) ENGINE=InnoDB AUTO_INCREMENT=502161227 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `SensorTemperature`
--

DROP TABLE IF EXISTS `SensorTemperature`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `SensorTemperature` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `value` float NOT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `device_id_2` (`device_id`,`created_on`),
  CONSTRAINT `_SensorTemperature_ibfk_1` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`)
) ENGINE=InnoDB AUTO_INCREMENT=507184889 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `TempUserDeviceList`
--

DROP TABLE IF EXISTS `TempUserDeviceList`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `TempUserDeviceList` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `role` enum('SuperAdmin','Admin','Guest','') COLLATE utf8_unicode_ci NOT NULL DEFAULT 'Guest',
  PRIMARY KEY (`row_id`),
  UNIQUE KEY `uk_user_device` (`user_id`,`device_id`),
  KEY `user_id` (`user_id`),
  KEY `device_id` (`device_id`),
  CONSTRAINT `TempUserDeviceList_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `User` (`user_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `TempUserDeviceList_ibfk_2` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`)
) ENGINE=InnoDB AUTO_INCREMENT=491 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `User`
--

DROP TABLE IF EXISTS `User`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `User` (
  `user_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `created_on` datetime NOT NULL,
  `first_name` varchar(30) COLLATE utf8_unicode_ci DEFAULT NULL,
  `last_name` varchar(30) COLLATE utf8_unicode_ci DEFAULT NULL,
  `email` varchar(50) COLLATE utf8_unicode_ci NOT NULL,
  `status` enum('Pending','Active','Closed','Banned') COLLATE utf8_unicode_ci NOT NULL,
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `UserCheckin`
--

DROP TABLE IF EXISTS `UserCheckin`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `UserCheckin` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `latitude` double NOT NULL,
  `longitude` double NOT NULL,
  `altitude` double NOT NULL,
  `accuracy` double NOT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `user_id_row_id` (`user_id`,`row_id`) COMMENT 'optimize group by user query',
  CONSTRAINT `UserCheckin_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `User` (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=1104194 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `UserCredential`
--

DROP TABLE IF EXISTS `UserCredential`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `UserCredential` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `credential` varchar(60) COLLATE utf8_unicode_ci NOT NULL,
  PRIMARY KEY (`row_id`),
  UNIQUE KEY `user_id` (`user_id`),
  CONSTRAINT `UserCredential_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `User` (`user_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=3716 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `UserDeviceHistory`
--

DROP TABLE IF EXISTS `UserDeviceHistory`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `UserDeviceHistory` (
  `user_device_history_id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `start` datetime NOT NULL,
  `end` datetime DEFAULT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`user_device_history_id`),
  KEY `user_id` (`user_id`),
  KEY `device_id` (`device_id`),
  CONSTRAINT `UserDeviceHistory_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `User` (`user_id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `UserDeviceHistory_ibfk_2` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB AUTO_INCREMENT=2151 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `UserDeviceList`
--

DROP TABLE IF EXISTS `UserDeviceList`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `UserDeviceList` (
  `row_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `role` enum('SuperAdmin','Admin','Guest','') COLLATE utf8_unicode_ci NOT NULL DEFAULT 'Guest',
  PRIMARY KEY (`row_id`),
  UNIQUE KEY `uk_user_device` (`user_id`,`device_id`),
  KEY `user_id` (`user_id`),
  KEY `device_id` (`device_id`),
  CONSTRAINT `UserDeviceList_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `User` (`user_id`),
  CONSTRAINT `UserDeviceList_ibfk_2` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`)
) ENGINE=InnoDB AUTO_INCREMENT=3613 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `UserFeedback`
--

DROP TABLE IF EXISTS `UserFeedback`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `UserFeedback` (
  `row_id` int(11) NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `user_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `feedback` float NOT NULL,
  `created_on` datetime NOT NULL,
  `control_target` enum('Ambi','Away_Humidity_Upper','Humidex','Humidity','Temperature','Manual','Away_Temperature_Upper','Away_Humidex_Upper','Off','Away_Temperature_Lower','Away_Humidex_Lower','Away_Humidity_Lower','Climate') COLLATE utf8_unicode_ci DEFAULT NULL,
  PRIMARY KEY (`row_id`),
  KEY `device_id` (`device_id`),
  KEY `user_id` (`user_id`),
  KEY `device_id_2` (`device_id`,`created_on`),
  KEY `device_id_row_id` (`device_id`,`row_id`),
  CONSTRAINT `UserFeedback_ibfk_1` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `UserFeedback_ibfk_2` FOREIGN KEY (`user_id`) REFERENCES `User` (`user_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=60645 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ModeFeedback`
--

DROP TABLE IF EXISTS `ModeFeedback`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ModeFeedback` (
  `row_id` int(11) NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `user_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `mode_feedback` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`row_id`),
  KEY `device_id` (`device_id`),
  KEY `user_id` (`user_id`),
  KEY `device_id_2` (`device_id`,`created_on`),
  KEY `device_id_row_id` (`device_id`,`row_id`),
  CONSTRAINT `ModeFeedback_ibfk_1` FOREIGN KEY (`device_id`) REFERENCES `Device` (`device_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `ModeFeedback_ibfk_2` FOREIGN KEY (`user_id`) REFERENCES `User` (`user_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=60645 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Weather`
--

DROP TABLE IF EXISTS `Weather`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Weather` (
  `weather_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `location_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `temperature` float DEFAULT NULL,
  `humidity` float DEFAULT NULL,
  `date` datetime NOT NULL,
  PRIMARY KEY (`weather_id`),
  KEY `location_id` (`location_id`),
  KEY `location_id_2` (`location_id`,`date`),
  CONSTRAINT `Weather_ibfk_1` FOREIGN KEY (`location_id`) REFERENCES `Location` (`location_id`)
) ENGINE=InnoDB AUTO_INCREMENT=3037112 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `WeatherAPI`
--

DROP TABLE IF EXISTS `WeatherAPI`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `WeatherAPI` (
  `weather_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `location_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL,
  `timestamp` datetime NOT NULL,
  `precip_intensity` float DEFAULT NULL,
  `precip_probability` float DEFAULT NULL,
  `precip_type` varchar(5) COLLATE utf8_unicode_ci DEFAULT NULL,
  `temperature` float DEFAULT NULL,
  `apparent_temperature` float DEFAULT NULL,
  `dew_point` float DEFAULT NULL,
  `humidity` float DEFAULT NULL,
  `wind_speed` float DEFAULT NULL,
  `wind_bearing` float DEFAULT NULL,
  `visibility` float DEFAULT NULL,
  `cloud_cover` float DEFAULT NULL,
  `pressure` float DEFAULT NULL,
  `ozone` float DEFAULT NULL,
  PRIMARY KEY (`weather_id`),
  UNIQUE KEY `location_id` (`location_id`,`timestamp`),
  KEY `timestamp` (`timestamp`),
  CONSTRAINT `WeatherAPI_ibfk_1` FOREIGN KEY (`location_id`) REFERENCES `Location` (`location_id`)
) ENGINE=InnoDB AUTO_INCREMENT=5084776 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `DeviceModePreference`
--

DROP TABLE IF EXISTS `DeviceModePreference`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `DeviceModePreference` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `user_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `cool` tinyint(1) unsigned NOT NULL,
  `fan` tinyint(1) unsigned NOT NULL,
  `dry` tinyint(1) unsigned NOT NULL,
  `heat` tinyint(1) unsigned NOT NULL,
  `auto` tinyint(1) unsigned NOT NULL,
  `quantity` varchar(24) COLLATE utf8_unicode_ci DEFAULT '',
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`id`),
  KEY `device_quantity_created_on` (`device_id`,`quantity`,`created_on`),
  KEY `latest_by_device_quantity` (`device_id`,`quantity`,`id`)
) ENGINE=InnoDB AUTO_INCREMENT=27336 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `WiFiQuality`
--

DROP TABLE IF EXISTS `WiFiQuality`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `WiFiQuality` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `device_id` varchar(36) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `quality` tinyint(3) unsigned NOT NULL,
  `created_on` datetime NOT NULL,
  PRIMARY KEY (`id`),
  KEY `device_id_created_on` (`device_id`,`created_on`)
) ENGINE=InnoDB AUTO_INCREMENT=33541508 DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2016-07-13 15:32:54

--
-- Table structure for table `DaikinApplianceState`
--

DROP TABLE IF EXISTS `DaikinApplianceState`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `DaikinApplianceState` (
  `appliance_state_id` int(11) NOT NULL,
  `ventilation_state_id` int(11) NOT NULL,
  UNIQUE KEY `appliance_ventilation_state_id_unique_index` (`appliance_state_id`,`ventilation_state_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
--
-- Table structure for table `VentilationState`
--

DROP TABLE IF EXISTS `VentilationState`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `VentilationState` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `value` varchar(36) COLLATE utf8mb4_unicode_ci NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

