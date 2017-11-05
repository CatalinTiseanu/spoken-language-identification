# general packages
sudo apt-get update
sudo apt-get install -y g++
sudo apt-get install -y python-pip
sudo apt-get install -y python-numpy
sudo apt-get install -y git
sudo apt-get install -y unzip

# sox
sudo apt-get install -y sox libsox-fmt-mp3

# sphinx base utils for feature extraction
sudo apt-get install -y sphinxbase-utils

# em4gmm
git clone https://github.com/juandavm/em4gmm
cd em4gmm

sudo apt-get install -y zlib1g-dev
sudo make
sudo make install

cd ..

# scikit-learn requirements
sudo apt-get install -y build-essential python-dev python-setuptools \
                     python-numpy python-scipy \
                     libatlas-dev libatlas3gf-base

# pip packages
pip install joblib
pip install pandas tqdm
pip install -U scikit-learn

