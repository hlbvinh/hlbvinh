#!/usr/bin/env python
import glob
from setuptools import setup, find_packages

MAJOR = 0
MINOR = 1
PATCH = 0

packages = find_packages()

setup(
    name="skynet",
    version="{}.{}.{}".format(MAJOR, MINOR, PATCH),
    description="Ambi Climate Machine Learning Components",
    author="Mathis Antony",
    author_email="mathis@ambiclimate.com",
    packages=packages,
    entry_points={"console_scripts": ["skynet = scripts.__main__:main"]},
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    data_files=[("airflow/dags", glob.glob("airflow/dags/*.py"))],
    include_package_data=True,
)
