# Erasure coding for distributed matrix multiplication
Distributed matrix multiplication is widely used in several scientific domains. It is well recognized that computation times on distributed clusters are often dominated by the slowest workers (called stragglers). The stragglers are treated as erasures in this implementation. The computation can be completed as long as a certain number of workers (called the recovery threshold) complete their assigned tasks. We present a novel coding strategy for this problem when the absolute values of the matrix entries are sufficiently small. We demonstrate a tradeoff between the assumed absolute value bounds on the matrix entries and the recovery threshold. At one extreme, we are optimal with respect to the recovery threshold and on the other extreme, we match the threshold of prior work. 

## Requirements
- Python 2 (tested with version 2.7.12)
- Open MPI (tested with version 2.1.2)
- MPI4Py (tested with version 3.0.0)

## Execution instructions
### Parameters
You should carefully set the parameters in `proposedMM.py`. The most basic ones are:
- `N`: number of MPI server processes (not including the master)
- `m == n == p == 2`
- `loadAB`: if set to 1 matrices `../pregenerateAB/A.npy` and `../pregenerateAB/B.npy` of dimension `s x r` and `s x t`, respectively, need to be existent so that they are loaded into our program. If set to 0, new matrices A, B of these dimensions will be generated.
Other parameters may need to be adjusted (see [paper](https://ieeexplore.ieee.org/document/8528366)). The Open MPI hostfile of the master machine should contain the hostnames of the computing servers or you can manually specify a hostfile when running the code.  

### Files and file paths
If you want to use pregenerated matrices, they should be in `/pregenerateAB` folder and have appropriate dimensions (see above).

### Execution
Determine the location of the Open MPI hostfile you will be using, say `/etc/openmpi/openmpi-default-hostfile`.

Then, run `mpirun -mca btl ^openib --hostfile /etc/openmpi/openmpi-default-hostfile python2 proposedMM.py` to execute the algorithm.

## Citation
If you use this code, please cite our IEEE Communications Letters [paper](https://ieeexplore.ieee.org/document/8528366):

  
```
@ARTICLE{8528366, 
author={L. Tang and K. Konstantinidis and A. Ramamoorthy}, 
journal={IEEE Communications Letters}, 
title={Erasure Coding for Distributed Matrix Multiplication for Matrices With Bounded Entries}, 
year={2019}, 
volume={23}, 
number={1}, 
pages={8-11}, 
keywords={Interference;Manganese;Encoding;Matrix decomposition;Fault tolerance;Fault tolerant systems;Interpolation;Distributed computing;erasure codes;stragglers}, 
doi={10.1109/LCOMM.2018.2880213}, 
ISSN={1089-7798}, 
month={Jan},}
```