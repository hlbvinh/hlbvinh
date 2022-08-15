"""MySQL database queries for interface with AmbiNet DB."""
import functools
from contextlib import closing
from datetime import datetime
from typing import Optional, Tuple

import pymysql
import pytz
import rapidjson as json

from ..log_util import get_logger
from .dbconnection import Pool

log = get_logger(__name__)
WEATHER_API_TABLE = "WeatherAPI"


def commit_wrap(func):
    @functools.wraps(func)
    def wrap(dbcon, *args, **kwargs):
        try:
            ret = func(dbcon, *args, **kwargs)
            dbcon.db.commit()
        except pymysql.err.Error as e:
            log.exception(e)
            dbcon.db.rollback()
            return None

        return ret

    return wrap


@commit_wrap
def execute(dbcon, query, args=tuple(), retry=True):

    with closing(dbcon.cursor()) as cursor:
        try:
            r = cursor.execute(query, args)  # pylint:disable=no-member
        except pymysql.err.Error as exc:
            log.exception(exc)
            if retry:
                log.debug("retrying query {} with parameters {}" "".format(query, args))
                try:
                    dbcon.db.ping()
                    return execute(dbcon, query, args, retry=False)
                except pymysql.err.Error as exc:
                    log.error(f"can't ping db, giving up: {exc}")
                    return ()
            raise

        if r:
            return cursor.fetchall()  # pylint:disable=no-member
        return ()


def commit(dbcon):
    return dbcon.db.commit()


def rollback(dbcon):
    return dbcon.db.rollback()


def insert(dbcon, string, args=None):
    """Insert records.

    Use only for INSERT Queries. Commit/rollback if execute returns True/False
    value.

    """
    with closing(dbcon.cursor()) as cursor:
        ret = cursor.execute(string, args)  # pylint:disable=no-member
        if ret:
            commit(dbcon)
        else:
            rollback(dbcon)
        return ret


def get(
    dbcon,
    database,
    table,
    columns,
    conditions,
    order_column="row_id",
    order="ASC",
    N=None,
):
    """Fetch our cached weather API Data.

    The location based binning is only done in a very rough way
    by using a fixed tolerance on lat/lon.

    Parameters
    ----------
    timestamp: datetime
    database: str
    table: str
    """

    if isinstance(columns, str):
        columns = [columns]

    sql = """
    SELECT
        {columns}
    FROM
        {database}.{table}
    """.format(
        columns=", ".join(columns), table=table, database=database
    )

    if not conditions:
        raise ValueError("empty argument conditions not supported")

    cond = "WHERE " + " AND ".join(["{0}=%({0})s".format(k) for k in conditions])
    params = conditions.copy()

    order = " ORDER BY {} {} ".format(order_column, order)

    if N is not None:
        limit = " LIMIT %(N)s"
        params["N"] = N
    else:
        limit = ""

    return execute(dbcon, sql + cond + order + limit, params)


async def get_appliance(pool: Pool, device_id: str) -> Optional[str]:
    resp = await pool.execute(
        "SELECT appliance_id FROM DeviceApplianceList " "WHERE device_id = %s",
        (device_id,),
    )
    if len(resp) >= 1:
        return resp[0]["appliance_id"]
    return None


def query_current_appliance_from_device(device_id):
    q = """
    SELECT appliance_id FROM DeviceApplianceHistory WHERE device_id = %s
    ORDER BY device_appliance_history_id DESC LIMIT 1"""
    return q, (device_id,)


def query_device_appliance_list():
    return "SELECT device_id, appliance_id FROM DeviceApplianceList", tuple()


def query_appliance_states(
    appliance_id,
    start,
    end,
    columns=[
        "appliance_state_id",
        "appliance_id",
        "temperature AS temperature_set",
        "LOWER(mode) AS mode",
        "LOWER(power) AS power",
        "LOWER(fan) AS fan",
        "created_on",
        "LOWER(origin) AS origin",
    ],
):
    start = start or datetime.fromtimestamp(0)
    end = end or datetime.utcnow()
    columns = ", ".join(columns)
    sql = """
    (SELECT
        {columns}
    FROM
        ApplianceState
    WHERE
        appliance_id = %(appliance_id)s
    AND
        created_on >= %(start)s
    AND
        created_on <= %(end)s)
    UNION
    (SELECT
        {columns}
    FROM
        ApplianceState
    WHERE
        appliance_id = %(appliance_id)s
    AND
        created_on < %(start)s
    ORDER BY
        created_on
    DESC LIMIT 1)
    ORDER BY created_on ASC""".format(
        columns=columns
    )
    return sql, {"appliance_id": appliance_id, "start": start, "end": end}


async def get_weather_api_data_from_device(
    pool,
    device_id,
    start,
    end,
    columns=[
        "apparent_temperature",
        "cloud_cover",
        "dew_point",
        "humidity",
        "precip_intensity",
        "precip_probability",
        "precip_type",
        "pressure",
        "ozone",
        "temperature",
        "timestamp",
        "visibility",
        "wind_bearing",
        "wind_speed",
    ],
    table=WEATHER_API_TABLE,
):

    location_id = await pool.execute(
        *query_location_from_device(device_id, ["location_id"])
    )

    if not location_id:
        return ()

    query, params = query_time_series(
        table,
        columns,
        {"location_id": location_id[0]["location_id"]},
        None,
        start,
        end,
        None,
        "timestamp",
        "ASC",
    )
    return await pool.execute(query, params)


def query_weather_from_device(device_id, start, end):
    sql = """
    SELECT
        W.temperature AS temperature_out,
        W.humidity * 100 AS humidity_out,
        W.timestamp AS timestamp
    FROM
        WeatherAPI W
    JOIN
        LocationDeviceList LD
    JOIN
    (
        SELECT
            MAX(row_id) AS row_id
        FROM
            LocationDeviceList
        WHERE
            device_id = %(device_id)s
    ) AS LD_last
    ON
        LD_last.row_id = LD.row_id
    AND
        LD.location_id = W.location_id
    AND
        W.timestamp >= %(start)s
    AND
        W.timestamp < %(end)s
    ORDER BY
        W.timestamp
    DESC"""
    return sql, {"device_id": device_id, "start": start, "end": end}


def query_location_from_device(device_id, columns=["location_id"]):
    if isinstance(columns, str):
        columns = [columns]
    columns = ", ".join(["L." + col for col in columns])
    sql = """
    SELECT
        {columns}
    FROM
        Location L
    INNER JOIN
        LocationDeviceList LD
    ON
        L.location_id = LD.location_id
    WHERE
        LD.row_id = (
        SELECT
            MAX(row_id) as row_id
        FROM
            LocationDeviceList
        WHERE
            device_id = %(device_id)s
    )
    """.format(
        columns=columns
    )
    return sql, {"device_id": device_id}


def get_location_from_device(dbcon, device_id, columns=["location_id"]):
    return execute(dbcon, *query_location_from_device(device_id, columns))


def get_device_from_location(dbcon, **kwargs):
    q = """
    SELECT
        D.device_id, D.location_id
    FROM
        LocationDeviceList D
    INNER JOIN
        Location L
    ON
        L.location_id = D.location_id
    INNER JOIN
    (SELECT
        MAX(row_id) as max_id
    FROM
        LocationDeviceList
    GROUP BY device_id
    ) D2
    ON
        D.row_id = D2.max_id
    """
    conditions = ["{0}=%({0})s".format(k) for k, v in kwargs.items()]
    if conditions:
        q += "WHERE " + " AND ".join(conditions)
    return execute(dbcon, q, kwargs)


def query_devices_from_location_id(location_id: str):
    sql = """
    SELECT
        D.device_id
    FROM
        LocationDeviceList D
    INNER JOIN
        Location L
    ON
        L.location_id = D.location_id
    INNER JOIN
    (SELECT
        MAX(row_id) as max_id
    FROM
        LocationDeviceList
    GROUP BY device_id
    ) D2
    ON
        D.row_id = D2.max_id
    WHERE
        D.location_id = %(location_id)s
    """
    return sql, {"location_id": location_id}


def update_device_location(dbcon, location_id, latitude, longitude):
    q = """
    UPDATE
        Location
    SET
        latitude = %s,
        longitude = %s
    WHERE
        location_id = %s
    """
    return execute(dbcon, q, (latitude, longitude, location_id))


def query_appliance_states_from_device(
    device_id,
    start=None,
    end=None,
    columns=[
        "appliance_state_id",
        "appliance_id",
        "temperature AS temperature_set",
        "mode",
        "power",
        "fan",
        "created_on",
        "origin",
    ],
):
    start = start or datetime.fromtimestamp(0)
    end = end or datetime.utcnow()
    columns = ", ".join(["A.{}".format(c) for c in columns])
    sql = """
    (
        SELECT {columns}
        FROM ApplianceState A
        WHERE A.appliance_id = (
            SELECT appliance_id
            FROM DeviceApplianceList
            WHERE device_id = %(device_id)s
            LIMIT 1
        )
        AND A.created_on < %(start)s
        ORDER BY A.created_on DESC
        LIMIT 1
    ) UNION (
        SELECT {columns}
        FROM ApplianceState A
        WHERE A.appliance_id = (
            SELECT appliance_id
            FROM DeviceApplianceList
            WHERE device_id = %(device_id)s
            LIMIT 1
        )
        AND A.created_on > %(start)s
        AND A.created_on < %(end)s
    )
    ORDER BY
        appliance_state_id
    """.format(
        columns=columns
    )
    return sql, {"device_id": device_id, "start": start, "end": end}


def get_appliance_states_from_device(dbcon, device_id, start=None, end=None):
    return execute(dbcon, *query_appliance_states_from_device(device_id, start, end))


def insert_ac_event_trigger(
    user_id, device_id, name, trigger_type, action, trigger_rule, enabled, created_on
):
    q = """
    INSERT INTO
        ACEventTrigger
        (trigger_id, user_id, device_id, name, trigger_type, action,
         created_on, trigger_rule, enabled)
    VALUES
        (NULL, %(user_id)s, %(device_id)s, %(name)s, %(trigger_type)s,
         %(action)s, %(created_on)s, %(trigger_rule)s, %(enabled)s)
    """
    return (
        q,
        {
            "user_id": user_id,
            "device_id": device_id,
            "name": name,
            "trigger_type": trigger_type,
            "action": json.dumps(action),
            "created_on": created_on,
            "trigger_rule": json.dumps(trigger_rule),
            "enabled": enabled,
        },
    )


def query_last_control_target(device_id):
    q = """
    SELECT
        device_id, LOWER(quantity) AS quantity, value,
        created_on,  LOWER(origin) AS origin
    FROM
        ApplianceControlTarget
    WHERE
        device_id = %s
    ORDER BY
        created_on
    DESC
        LIMIT 1
    """
    return q, (device_id,)


def query_last_appliance_state(device_id: str):
    q = """
    SELECT
        A.appliance_id,
        LOWER(A.power) as power,
        LOWER(A.mode) as mode,
        LOWER(A.fan) as fan,
        LOWER(A.louver) as louver,
        LOWER(A.swing) as swing,
        A.temperature,
        A.created_on,
        LOWER(A.origin) as origin,
        (SELECT value FROM VentilationState WHERE id=DVS.ventilation_state_id) ventilation
    FROM (
        SELECT max(A.appliance_state_id) appliance_state_id
        FROM ApplianceState A
        JOIN DeviceApplianceList DAL ON DAL.appliance_id = A.appliance_id
        WHERE DAL.device_id = %s
    ) TEMP
    JOIN ApplianceState A ON A.appliance_state_id = TEMP.appliance_state_id
    LEFT OUTER JOIN DaikinApplianceState DVS ON DVS.appliance_state_id = A.appliance_state_id
    """
    return q, (device_id,)


def query_last_on_appliance_state(device_id: str):
    q = """
    SELECT
        A.appliance_id,
        LOWER(A.power) as power,
        LOWER(A.mode) as mode,
        LOWER(A.fan) as fan,
        LOWER(A.louver) as louver,
        LOWER(A.swing) as swing,
        A.temperature,
        A.created_on,
        LOWER(A.origin) as origin,
        (SELECT value FROM VentilationState WHERE id=DVS.ventilation_state_id) ventilation
    FROM (
        SELECT max(A.appliance_state_id) appliance_state_id
        FROM ApplianceState A
        JOIN DeviceApplianceList DAL ON DAL.appliance_id = A.appliance_id
        WHERE DAL.device_id = %s
        AND A.power = 'On'
    ) TEMP
    JOIN ApplianceState A ON A.appliance_state_id = TEMP.appliance_state_id
    LEFT OUTER JOIN DaikinApplianceState DVS ON DVS.appliance_state_id = A.appliance_state_id
    """
    return q, (device_id,)


def query_last_mode_preferences(device_id: str):
    q = """
    SELECT
        quantity, cool, heat, dry, fan, auto, created_on
    FROM
        DeviceModePreference A
    JOIN (
        SELECT
            MAX(id) as id
        FROM
            DeviceModePreference
        WHERE
            device_id = %s
        GROUP BY quantity
    ) B
    USING (id)
    """
    return q, (device_id,)


def query_control_targets(
    device_id, start, end, columns=["device_id", "quantity", "value", "created_on"]
):
    columns = ", ".join(columns)
    sql = """
    (SELECT
        {columns}
    FROM
        ApplianceControlTarget
    WHERE
        device_id = %(device_id)s
    AND
        created_on < %(start)s
    ORDER BY
        created_on DESC
    LIMIT 1)
    UNION
    (SELECT
        {columns}
    FROM
        ApplianceControlTarget
    WHERE
        device_id = %(device_id)s
    AND
        created_on BETWEEN %(start)s AND %(end)s
    ORDER BY
        created_on)
    """.format(
        columns=columns
    )
    return sql, {"device_id": device_id, "start": start, "end": end}


def query_time_series(
    table,
    columns,
    conditions,
    timestamp=None,
    start=None,
    end=None,
    N=None,
    timestamp_column="created_on",
    order="ASC",
):
    """Fetch our cached weather API Data.

    The location based binning is only done in a very rough way
    by using a fixed tolerance on lat/lon.

    Parameters
    ----------
    timestamp: datetime
    table: str
    """

    if isinstance(columns, str):
        columns = [columns]

    sql = """
    SELECT
        {columns}
    FROM
        {table}
    """.format(
        columns=", ".join(columns), table=table
    )

    if not conditions:
        raise ValueError("empty argument conditions not supported")

    cond = "WHERE " + " AND ".join(["{0}=%({0})s".format(k) for k in conditions])
    params = conditions.copy()

    if timestamp is not None:
        time_cond = " AND {} = %(timestamp)s".format(timestamp_column)
        params["timestamp"] = timestamp
    elif start is not None and end is not None:
        time_cond = " AND {0} > %(start)s AND {0} <= %(end)s".format(timestamp_column)
        params["start"] = start
        params["end"] = end
    elif start is not None:
        time_cond = " AND {} > %(start)s".format(timestamp_column)
        params["start"] = start
    elif end is not None:
        time_cond = " AND {} <= %(end)s".format(timestamp_column)
        params["end"] = end
    else:
        time_cond = " "

    order = " ORDER BY {} {} ".format(timestamp_column, order)

    if N is not None:
        limit = " LIMIT %(N)s"
        params["N"] = N
    else:
        limit = ""

    return sql + cond + time_cond + order + limit, params


def query_weather_interval(location_id, start, end, hours=4):
    sql = """
    SELECT
        MIN(CONVERT(DATE_FORMAT(timestamp,'%%Y-%%m-%%d %%H'),DATETIME))

            AS timestamp,
        AVG(temperature) AS temperature,
        100.0 * AVG(humidity) AS humidity
    FROM
        WeatherAPI W
    WHERE
        location_id = %(location_id)s
    AND
        W.timestamp BETWEEN %(start)s and %(end)s
    GROUP BY
        DATE(W.timestamp), FLOOR(HOUR(W.timestamp) / %(hours)s)
    ORDER BY
        timestamp
    """
    return sql, {"location_id": location_id, "start": start, "end": end, "hours": hours}


def query_weather(location_id, timestamp, N=24):
    return query_time_series(
        WEATHER_API_TABLE,
        columns=["temperature as temperature_out", "100 * humidity as humidity_out"],
        conditions={"location_id": location_id},
        end=timestamp,
        timestamp_column="timestamp",
        N=N,
        order="DESC",
    )


def query_user_feedback(
    device_id=None, user_id=None, timestamp=None, start=None, end=None
):
    conditions = {}
    if device_id is not None:
        conditions["device_id"] = device_id
    if user_id is not None:
        conditions["user_id"] = user_id

    if timestamp is not None:
        kwargs = {"end": timestamp, "N": 1, "order": "DESC"}
    else:
        kwargs = {}
        if start is not None:
            kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end

    return query_time_series(
        "UserFeedback",
        columns=["device_id", "user_id", "created_on", "feedback"],
        conditions=conditions,
        **kwargs,
    )


async def get_time_zone(pool, device_id):
    loc_zone = await pool.execute(
        """
    SELECT
        L.timezone
    FROM
        Location L
    INNER JOIN
        LocationDeviceList LD
    ON
        L.location_id = LD.location_id
    WHERE
        LD.row_id
    = (
        SELECT
            MAX(row_id) as row_id
        FROM
            LocationDeviceList
        WHERE
            device_id = %s
    )""",
        (device_id,),
    )
    zone = None
    if loc_zone:
        try:
            zone = pytz.timezone(loc_zone[0]["timezone"])
        except (pytz.UnknownTimeZoneError, AttributeError):
            pass

    if zone is None:
        dev_zone = await pool.execute(
            "SELECT timezone FROM Device WHERE device_id = %s", (device_id,)
        )
        if dev_zone:
            try:
                zone = pytz.timezone(dev_zone[0]["timezone"])
            except (pytz.UnknownTimeZoneError, AttributeError):
                pass

    if zone is None:
        return pytz.timezone("Asia/Hong_Kong")
    return zone


def query_device_timezone(device_id):
    q = """
    SELECT
        COALESCE(L.timezone, D.timezone, 'Asia/Hong_Kong') AS timezone
    FROM
        LocationDeviceList LD
    JOIN
        Location L
    JOIN
        Device D
    JOIN (
        SELECT
            device_id,
            MAX(row_id) AS row_id
        FROM
            LocationDeviceList
        WHERE
            device_id = %s
    ) AS LD_last
    ON
        LD_last.row_id = LD.row_id
    AND
        LD_last.device_id = LD.device_id
    AND
        LD_last.device_id = D.device_id
    AND
        L.location_id = LD.location_id
    """
    return q, (device_id,)


def query_timezone(location_id):
    q = """
    SELECT
        timezone
    FROM
        Location L
    WHERE
        location_id = %s
    """
    return q, (location_id,)


#
# XXX MATHIS 2016 05 20: Will soon no longer be correct because not all sensor
#                        data is in MySQL db.
#


def get_device_online_intervals(dbcon):
    sql = """
    SELECT
        device_id,
        min(created_on) AS start,
        max(created_on) AS end
    FROM
        SensorTemperature
    GROUP BY
        device_id
    """
    return execute(dbcon, sql)


def query_feedback_intervals():
    sql = """
    SELECT
        device_id,
        min(created_on) AS start,
        max(created_on) AS end
    FROM
        UserFeedback
    GROUP BY
        device_id
    """
    return sql, tuple()


def query_last_mode_feedback(device_id):
    q = """
    SELECT
        device_id, user_id, mode_feedback, created_on
    FROM
        ModeFeedback
    WHERE
        device_id = %s
    ORDER BY
        created_on
    DESC LIMIT 1
    """
    return q, (device_id,)


def query_latest_feedbacks(device_id: str) -> Tuple[str, Tuple[str]]:
    q = """
    SELECT
        uf.user_id, uf.feedback, uf.created_on
    FROM
        UserFeedback AS uf
    JOIN
        (SELECT
            MAX(row_id) AS row_id
        FROM
            UserFeedback
        WHERE
            device_id = %s
        GROUP BY
            user_id) AS uv
        ON
            uf.row_id = uv.row_id
    """
    return q, (device_id,)
