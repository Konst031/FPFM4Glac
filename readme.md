# A FEniCSx based Fracture Phase-Field model for Glaciers
This repository is associated with the paper "Fracture Phase-Field Modeling of Supraglacial Lake
Drainage and Marine Ice Cliff Instability"

## Requirements
Built for fenics-dolfinx version 0.9.0.
We reccomend the usage of an appropriate Docker image from:
https://hub.docker.com/r/dolfinx/dolfinx/tags 

## Quickstart
1. create a mesh using an appropriate mesh generation script (Parameters for geometries can be altered in the scripts)
2. adjust the filepaths in crackMain.py if necessary
3. Choose simulation parameters in the Testsuite.csv file:
    obligatory arguments: meshInputFile, bcType; optionl arguments can be found in source/validKeys.py; default values can be found in the class definitions within the source files
4. run a simulation via: 
    ```console
    python3 crackMain.py JOB_NAME TESTSUITE_CSV_PATH
    ```    
5. visualize the result e.g. using paraview