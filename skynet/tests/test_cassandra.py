import pytest


def test_cassandra_connection(cassandra_session):
    res = cassandra_session.execute("SELECT dateof(now()) FROM system.local")
    assert res


@pytest.mark.asyncio
async def test_cassandra_async_connection(cassandra_session):
    res = await cassandra_session.execute_async(
        "SELECT dateof(now()) FROM system.local"
    )
    assert res[0]
