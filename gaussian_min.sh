#!/bin/bash
nvcc main.cu -I include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -arch=compute_50 -code=sm_50 -o build/gaussian

#printf "randgentime;calctime;iterations_reachedn;Gpus;tile_width;iterations;\n"
for gpu_n in 1 ; do
    for cpu_p in 0.00; do
	for iterations in 5 10; do
	    for kw in 10 20; do
                for tile_width in 8 16 32; do
	            mpirun -np 1 build/gaussian $gpu_n 1 $cpu_p $tile_width $iterations 1 $kw
	        done
                mpirun -np 1 build/gaussian $gpu_n 1 $cpu_p 12 $iterations 0 $kw
    	    done
        done
    done
done
