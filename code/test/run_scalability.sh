#!/bin/bash

echo "Compiling"
make

if [ $? -ne 0 ]; then
    echo "Error in 'make'."
    exit 1
fi

echo "=========================================="
echo "Scaling Test (MPI)"
echo "=========================================="

for procs in 1 2 4 8
do
    export OMP_NUM_THREADS=1

    mpirun -n $procs ./run_test
    
    echo "--------------------------------------------------"
done

echo ""
echo "=========================================="
echo "Test completed! Check the "data" folder."
echo "=========================================="