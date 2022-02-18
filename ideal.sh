#!/bin/bash
nvcc main.cu -I include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -arch=compute_50 -code=sm_50 -o build/gaussian
for sm in 1 2; do
	for blockm in 1 2 3 4 5 67 8; do
		for kw in 8 10 12 14 16 18 20; do
			for tw in 16 32; do
	    			build/gaussian 1 1 0.0 $tw 1 $kw $blockm $sm
	    			wait
			done
		done
	done
done
