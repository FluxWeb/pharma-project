#!/bin/bash
rm -rf ../build/* ../bin/* ../lib/*
cd ../build
cmake ..
make
cd ../
./bin/pharma_project
cd ../scripts/
