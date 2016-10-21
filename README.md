This is a currently ongoing implementation of Structure From Motion pipeline on HPC cluster using Boost MPI.

##Required modules:
1. OpenCV
2. MPI (currently tested for OpenMPI)
3. Boost library built with filesystem, mpi and serialization components.
4. Flann 1.8.4 (Link: http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann-1.8.4-src.zip )

##Running Examples:

Since I have added some images alongside (which I know is not a good thing), you can get started building and running the code in following steps.
1. $ mkdir build
2. $ cd build
3. $ cmake ..
4. $ make
5. $ cd ..
6. $ mpirun -n 4 ./bin/key_demo
7. $ mpirun -n 4 ./bin/match_demo

The development of this library is still in process. Please point out the inefficiencies in the issue tracker and I will try to resolve them. Pull Requests are most welcome!
