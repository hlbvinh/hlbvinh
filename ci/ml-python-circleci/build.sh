#/bin/bash
set -e

if [ -z "$CI" ]; then
    MIRROR="http://ftp.cuhk.edu.hk/pub/Linux/ubuntu/"
else
    MIRROR="http://us-east-1.ec2.archive.ubuntu.com/ubuntu"
fi
sed -i "s%http://archive.ubuntu.com/ubuntu/%$MIRROR%g" /etc/apt/sources.list

# purge old data
# rm -rf /var/lib/apt/lists/*

apt-get update
apt-get install -y software-properties-common

# Add repo with python
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get upgrade -y

apt-get -y install build-essential \
                   curl \
                   git \
                   libssl-dev \
                   libffi-dev \
                   mysql-client \
                   python3.6-dev \
                   sudo

# install npm and jscpd for code duplication detection
curl -sL https://deb.nodesource.com/setup_8.x | bash
apt-get -y install nodejs
npm install jscpd
npm cache clean --force

# match circleci image sudo configuration
useradd -m -s /bin/bash circleci
echo 'circleci ALL=NOPASSWD: ALL' > /etc/sudoers.d/50-circleci
echo 'Defaults    env_keep += "DEBIAN_FRONTEND"' > /etc/sudoers.d/env_keep

# get pip
curl https://bootstrap.pypa.io/get-pip.py | python3.6 - '--no-setuptools' '--no-wheel'


# clean apt-get
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache
