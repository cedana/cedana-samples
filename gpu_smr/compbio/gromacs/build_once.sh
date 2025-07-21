#!/usr/bin/env sh

# Run once to build, used for embedding binary inside container
# This script builds GROMACS from source and installs it. If running on your own machine, should build.
wget https://ftp.gromacs.org/gromacs/gromacs-2025.2.tar.gz
tar xfz gromacs-2025.2.tar.gz
cd gromacs-2025.2
mkdir build
cd build
cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON
make
make check
sudo make install
