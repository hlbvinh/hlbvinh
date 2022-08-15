from ..utils import weather


def test_classify_outdoor_temperature():
    pairs = [
        (0, "cold"),
        (13.99, "cold"),
        (14, "cool"),
        (21.99, "cool"),
        (22, "warm"),
        (29.99, "warm"),
        (30, "hot"),
    ]
    for temperature, condition in pairs:
        assert weather.classify_outdoor_temperature(temperature) == condition


def test_classify_outdoor_humidity():
    pairs = [(0, "dry"), (64.99, "dry"), (65, "humid"), (100, "humid")]
    for humidity, condition in pairs:
        assert weather.classify_outdoor_humidity(humidity) == condition
