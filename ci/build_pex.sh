#!/bin/bash
set -e

# update build dependencies inside docker only
if [ -f /.dockerenv ]; then
    sudo pip install $(cat requirements.txt | grep 'pex==')
fi

# pex doesn't support git resolvables
if [ -z ${GITHUB_TOKEN+x} ]; then
    echo "using github via ssh"
    GITURL=git@github.com:ambilabs/ambi_utils
else
    echo "using GITHUB_TOKEN"
    GITURL=https://$GITHUB_TOKEN@github.com/ambilabs/ambi_utils.git
fi

git clone $GITURL -b asyncio-plus ambi_utils || \
    echo "didn't clone ambi_utils, remove ./ambi_utils if you want to clone again"
TORCH_VERSION=`cat requirements.txt | grep -Po '(?<=torch==).*'`
PYTHON_VERSION=`cat tox.ini | grep -Po '(?<=basepython = python)[0-9]\.[0-9]' | tr -d .`
sed -e "/ambi_utils/c\ambi_utils" \
    -e "/torch==/c$(echo https://download.pytorch.org/whl/cpu/torch-${TORCH_VERSION}%2Bcpu-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}m-linux_x86_64.whl)" \
    requirements.txt > /tmp/requirements.txt

# for now we fix all dependencies to make sure that pex building is reproduceable and not influenced
# the content of CircleCI cache
cat subdependencies_requirements >> /tmp/requirements.txt

echo "### PEX BUILD DIR BEFORE BUILD ###"
ls -lah ~/.pex/build || true

# cache for max 1 year
# TODO once we have a lock file for the dependencies and subdependencies
# we could use the --intransitive option
pex -vvvv --cache-ttl 31556926 . -r /tmp/requirements.txt -c skynet -o python.pex

PEX_NAME="$(bash ci/get_pex_name.sh get_pex_name)"

echo "copying python.pex to $PEX_NAME"
mkdir -p ~/artifacts
cp python.pex ~/artifacts/$PEX_NAME

#echo "### PEX BUILD DIR AFTER BUILD ###"
#ls -lah ~/.pex/build || true
