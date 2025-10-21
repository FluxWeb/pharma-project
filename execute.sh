#!/bin/bash

rm build -r
mkdir build
cd build
cmake ..
make 
./bin/pharma_project