#!/usr/bin/env sh

#!/usr/bin/env sh
set -eux

# Build and install GROMACS system-wide
wget https://ftp.gromacs.org/gromacs/gromacs-2025.2.tar.gz
tar xfz gromacs-2025.2.tar.gz
cd gromacs-2025.2
mkdir build && cd build

cmake .. \
    -DGMX_BUILD_OWN_FFTW=ON \
    -DREGRESSIONTEST_DOWNLOAD=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local

make -j"$(nproc)"
make check
make install

# Ensure runtime linker finds the GROMACS shared libraries
echo "/usr/local/lib" >/etc/ld.so.conf.d/gromacs.conf
ldconfig

# Make gmx binary easily available
cp /usr/local/bin/gmx /usr/local/bin/gmx
