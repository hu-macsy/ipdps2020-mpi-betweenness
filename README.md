# ipdps2020-mpi-betweenness

This repository contains the necessary code + helper scripts to recreate the experiments used in "Scaling Betweenness Approximation to Billions of Edges by MPI-based Adaptive Sampling" (A. van der Grinten, H. Meyerhenke).

### Requirements:

- C++ compiler, `g++` 9.0 or higher recommended
- MPI-runtime, code was tested with OpenMPI 4 and MPICH
- `simexpal` (see Notes for more details about usage)

### Notes:

- To build the necessary code and run the tests use simexpal: https://github.com/hu-macsy/simexpal/. The configuration of the experiments can be found inside `experiments.yml`. 
- Instances are referenced by their names and path. Some of the smaller (both generated and real networks) instances can be downloaded here: https://box.hu-berlin.de/d/f2b95b8f887e4847a114/
- If other instances should be tested, they need to be converted to `NetworKitBinary` format. See here for more information: https://networkit.github.io/dev-docs/notebooks/IONotebook.html#NetworkitBinaryGraph-file-format You can also get in touch with macsy@informatik.hu-berlin.de if there need for a particular instance.

### Results:

- If you are just interest in the results, the subfolder `results` contains an archive (`data.tar.gz`) with output data from all experimental runs and several Jupyter notebooks. In the notebooks you can find the evaluation scripts, used for creating plots and table data for the paper.
