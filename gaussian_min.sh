#!/bin/bash
nvcc main.cu -I include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -arch=compute_50 -code=sm_50 -o build/gaussian

#printf "randgentime;calctime;iterations_reachedn;Gpus;tile_width;iterations;\n"
for gpu_n in 1; do
    for cpu_p in 0.00; do
	for iterations in 1; do
	    for kw in 10; do
		for block_mult in 125 250; do
                    for tile_width in 32; do
	                build/gaussian $gpu_n 1 $cpu_p $tile_width $iterations 1 $kw $block_mult
	            done
	        done
                #mpirun -np 1 build/gaussian $gpu_n 1 $cpu_p 12 $iterations 0 $kw 2
    	    done
        done
    done
done
