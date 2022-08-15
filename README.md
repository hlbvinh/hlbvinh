[![codecov.io](http://codecov.io/github/AmbiLabs/ambi_brain/coverage.svg?token=U7you7PjRd&branch=master)](http://codecov.io/github/AmbiLabs/ambi_brain?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

    ┌─┐┬┌─┬ ┬┌┐┌┌─┐┌┬┐
    └─┐├┴┐└┬┘│││├┤  │
    └─┘┴ ┴ ┴ ┘└┘└─┘ ┴
Machine Learning Components. Predictions. Estimations. Control.

### Architecture

check architecture.org for an overview of the codebase architecture.

### Develop
The quickest way to hack is to install with `pip` in editable mode.

```
pip install -e .
skynet --help
skynet user_model --help
```
Changes made in the repo are now directly visible in the package installed
via `pip`.

You need to install mysql, mongodb, cassandra and redis. You can install these
services to install these in docker then load data to myql and cassandra.

    install/systemd/install_docker_container.sh

Run the following script to copy unit files of database service to
etc/systemd/system, reload daemons, and start and enable services (sudo might be needed):

    install/systemd/enable_database_services.sh

To run tests,
```bash
tox -e lint         # lint changes with respect to master branch
tox -e lint -- all  # lint everything
tox -e local        # run required tests locally
tox                 # default: lint changes and run required tests locally
tox -r              # use -r to reload dependencies from requirements.txt before running tests.
```
To run only selected tests anything passed to tox is passed to pytest

    tox -e local -- skynet/tests/test_misc.py
    
We use black to automatically format code
```
pip install pre-commit
pre-commit install
```
This should setup git hooks to run black on the file before the commit

### Deploy
`$ENV` is either `staging` or `production`. `$GROUP` is `train` or `pred`.

```bash
fab $ENV $GROUP deploy:build_num=3824  # download/install pex and create app releases and airflow dags
fab staging pred restart_all
```
for more fine-grained control
```bash
fab $ENV $GROUP install_pex:BUILD_NUM           # (default=latest build)
fab $ENV $GROUP release:SERVICE                 # (default=all services)
fab $ENV $GROUP airflow_dag_release:DAG_NAME    # (default=all dags)
fab $ENV $GROUP remove_extracted_pex_artifacts  # remove what PEX extracts
fab $ENV $GROUP clean:n_keep=5                  # remove PEX archives
```

### Directory Structure
A release for a service is made by creating a release directory `$RELEASE`
containing

    - a copy of the service config file `config.yml`
    - a link called `skynet.pex` to a PEX file

The current release directory `/opt/ai/services/$SERVICE_NAME/current` is then
symlinked to one of these releases and its path is used to run the service.

```bash
PEX=/opt/ai/shared/pex/$RELEASE_TAG.pex
HOME=/opt/ai/services/$SERVICE_NAME
RELEASE=/opt/ai/services/$SERVICE_NAME/releases/[RELEASE_TAG]
/opt/ai/services/$SERVICE_NAME/current -> $RELEASE
/opt/ai/services/$SERVICE_NAME/current/skynet.pex -> $PEX
```

Similarly for airflow
```bash
/opt/ai/airflow/dags/$DAG_NAME/current ->
/opt/ai/airflow/$DAG_NAME/releases/$RELEASE_TAG/skynet.pex
                                                config.yml
                                                $DAG_NAME.py
                          config.yml

```
Inside the dags the executable path is `/opt/ai/airflow/dags/$DAG_NAME/current/skynet.pex`
