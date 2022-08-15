#!/bin/bash

echo "Hello, "$USER".  This script will get your mac ready for development. If homebrew is already installed then you need to read the script."
echo "Firstly, manually fork the ambi_brain repo on Github"

echo -n "Enter your github ID e.g. pallavbakshi and press [ENTER]: "
read GITHUBUSERNAME

/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

brew install python3

pip3 install virtualenv

mkdir ~/ambi
cd ~/ambi

#Generate and copy ssh keys and paste the ssh keys to github (paste manually)
ssh-keygen -b 4096
pbcopy < ~/.ssh/id_rsa.pub
ssh-add -K ~/.ssh/id_rsa

virtualenv -p python3 initambi

#Fork it on your personal github manually

git clone git@github.com:$GITHUBUSERNAME/ambi_brain

cd ~/ambi/ambi_brain

source ~/ambi/initambi/bin/activate

pip3 install -r requirements.txt

deactivate

read -n 1 -p "Paste the key to github. [ENTER] "

read -n 1 -p "Install docker manually. [ENTER] "

# Install MySQL client (Make sure it is only the client and not the server)
brew install mysql-client
echo 'export PATH="/usr/local/opt/mysql-client/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile

# Create docker containers and populate with data by running the script
# install/systemd/install_docker_container.sh

# Run tests
# Ignore LAPACK Warning

echo completed
