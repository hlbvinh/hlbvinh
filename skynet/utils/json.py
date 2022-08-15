import rapidjson as rj


def dumps(data):
    return rj.dumps(data, datetime_mode=rj.DM_ISO8601)


def loads(json):
    return rj.loads(json, datetime_mode=rj.DM_ISO8601)
