# ece1782project

## To compile CPU version
1. Enter the CPU directory and just run `make`
2. The output binary is called `particles`. Run it as `./particles`, specify the simulation you'd like to run and run it using "T"


## To compile GPU version
1. Run qmake inside the gpu directory
2. Compare the makefile created to the makefile.bk, make the changes so that they match (But ensure that the pathnames are local to your machine) \
TODO: \
Use environment variables to avoid this step
4. Run make and the binary is called `particles_cuda`
5. Save this makefile

