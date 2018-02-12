sudo apt-get install python3-pip zlib1g-dev -y
sudo pip3 install --upgrade pip
sudo pip3 install tensorflow==1.4.0 scipy sklearn

cd python
git clone https://github.com/liffiton/PyMiniSolvers.git
cd PyMiniSolvers
make
cd ../..

mkdir data dimacs snapshots
