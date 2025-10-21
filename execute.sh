#!/bin/bash

rm build -r
mkdir build
cd build
cmake ..
make 
cd ../
./bin/pharma_project