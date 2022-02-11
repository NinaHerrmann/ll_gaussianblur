#!/bin/bash
nvcc main.cu -I include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -arch=compute_50 -code=sm_50 -o build/gaussian

for kw in 10; do
	build/gaussian 1 1 0.0 16 1 $kw
done
