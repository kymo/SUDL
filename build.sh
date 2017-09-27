#!/usr/bin/env sh

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
