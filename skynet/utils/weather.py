def classify_outdoor_humidity(humidity: float) -> str:
    return "humid" if humidity >= 65.0 else "dry"


def classify_outdoor_temperature(temperature: float) -> str:
    if temperature < 14.0:
        return "cold"
    if temperature < 22.0:
        return "cool"
    if temperature < 30.0:
        return "warm"
    return "hot"


def classify_weather(temperature: float, humidity: float) -> str:
    return "{}_{}".format(
        classify_outdoor_temperature(temperature), classify_outdoor_humidity(humidity)
    )
