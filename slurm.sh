#!/bin/bash

#SBATCH --export=NONE
#SBATCH --partition=gpu2080
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --job-name=GaussianBlurLL_LL_REAL
#SBATCH --output=/scratch/tmp/n_herr03/gaussian/lowlevel/errorandoutput/Gaussian-1802-mili.txt
#SBATCH --error=/scratch/tmp/n_herr03/gaussian/lowlevel/errorandoutput/Gaussian-1802-mili.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n_herr03@uni-muenster.de

#module load intelcuda/2019a
#module load CMake/3.15.3
#module load CMake/3.15.3
module load palma/2020b
module load fosscuda/2020b
#module load gompic/2020b
#module load CMake/3.18.4
cd /home/n/n_herr03/gaussianblur_ll/

nvcc main.cu -I include/ -arch=compute_75 -code=sm_75 -o build/gaussian
#export OMP_NUM_THREADS=4

# vorl�ufig, bis MPI �ber Infiniband funktioniert
#export I_MPI_DEBUG=3
# export I_MPI_FABRICS=shm:ofa   nicht verf�gbar
# alternativ: Ethernet statt Infiniband:
#export I_MPI_FABRICS=shm:tcp

# mpirun /home/k/kuchen/Muesli4/build/$1 $2 $3
# parameters: array dim #MPI nodes
# mpirun /home/k/kuchen/Muesli4/build/da_test 32 2

# parameters: area size (needs to be quadratic) #MPI nodes
# mpirun /home/k/kuchen/Muesli4/build/mandelbrotDA 10000 2

# parameters: #processes (= dim of DA), #throws, #MPI nodes
# mpirun /home/k/kuchen/Muesli4/build/piDA 1000 1000000 2

# parameters: #DMCols #DMRows #nGPU #nRuns #CpuPercentage
for kw in 8 10 12 14 16 18 20; do
	for tile_width in 8 16 32; do
		/home/n/n_herr03/gaussianblur_ll/build/gaussian 1 1 0 $tile_width 1 $kw
	done
	/home/n/n_herr03/gaussianblur_ll/build/gaussian 1 1 0 12 1 $kw
done


#srun nvprof --analysis-metrics -o /scratch/tmp/e_zhup01/muesli-jacobi-analysis.%p.nvprof /home/e/e_zhup01/muesli4/build/jacobi -numdevices=1
# alternativ: mpirun -np 2 <Datei>
# alternativ: srun <Datei>
