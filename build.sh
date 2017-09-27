#!/usr/bin/env sh

# check cmake
cmake --version
if [$? -ne 0]; then
	echo 'cmake not found, please download in https://cmake.org/files/v3.0/cmake-3.0.2.tar.gz'
	exit 1
fi

if [ "build" ]; then  
	rm -rf build
fi
mkdir build  

if [ "output/bin" ]; then
	rm -rf output
fi
mkdir -p output/bin

cd build;
cmake ..
make -j 8
