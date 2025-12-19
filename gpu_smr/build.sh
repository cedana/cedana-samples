export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
find build -type f -executable -exec mv {} . \;
