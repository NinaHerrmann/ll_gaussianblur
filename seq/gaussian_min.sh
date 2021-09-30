#!/bin/bash
nvcc main.cu -I include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -o build/gaussian
#printf "randgentime;calctime;iterations_reachedn;Gpus;tile_width;iterations;\n"
for iterations in 1; do
    for kw in 2 4; do
        build/gaussian 0 1 0.00 12 $iterations 0 $kw
    done
done
