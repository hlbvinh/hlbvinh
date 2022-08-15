# pylint: skip-file
import os
import sys

# Mathis 2016 12 15
# This no longer works to run pytest with xdist (multi-cpu) plugin.
# It seems that skynet/tests/conftest.py gets loaded before pytest_configure
# is called which causes errors because we don't have the PEX environment
# activated.


def activate_pex(pex_path):
    sys.path.insert(0, os.path.abspath(os.path.join(pex_path, ".bootstrap")))
    from _pex import pex_bootstrapper

    pex_bootstrapper.bootstrap_pex_env(pex_path)


def pytest_configure(config):
    jobs = int(config.getoption("-n") or 0)
    pex_path = os.environ.get("PEXPATH")
    if pex_path and jobs > 0:
        activate_pex(pex_path)
        os.environ["PYTHONPATH"] = ":".join(sys.path)
