# ／usr/bin bash

git clone https://github.com/baidu/Paddle paddle
cd paddle


# swig
swig=swig.tar.gz
wget -O ${swig}  https://nchc.dl.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz
tar -vzxf ${swig}
cd swig
./configure --prefix=/usr/local/swig3.0.12
make
sudo make install

# wheel
sudo pip install wheel

# protobuf
protobuf=protobuf.tar.gz
wget -O ${protobuf}  https://github.com/google/protobuf/archive/v3.2.0.tar.gz
tar -vzxf ${protobuf}
cd protobuf
./configure
make
make check
sudo make install
cd ./python
python2 setup.py build
python2 setup.py test
python2 setup.py install




mkdir build && cd build
cmake .. -DWITH_GPU=OFF -DWITH_DOC=ON -DWITH_SWIG_PY=ON -WITH_GLOG=ON -WITH_GFLAGS=ON -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo make install
