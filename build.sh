#!/bin/bash
rm -rf build
mkdir build
cd build
cmake -DgenericIO_DIR=/home/supercomputes/apps/genericIO/lib -Dhypercube_DIR=/home/supercomputes/apps/hypercube/lib -DCMAKE_INSTALL_PREFIX=/home/supercomputes/apps/sandbox/Born3 ..
make
make install
