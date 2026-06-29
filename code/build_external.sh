# CCVT
rm -rf ccvt_mod/build
mkdir ccvt_mod/build/
cmake -Hccvt_mod -Bccvt_mod/build/ -DCMAKE_BUILD_TYPE=Release
cmake --build ccvt_mod/build --config Release

# janus
cd janus/
git clean -fx .
echo -e '[build-system]\nrequires = ["setuptools>=64", "wheel", "Cython"]\nbuild-backend = "setuptools.build_meta"' > pyproject.toml
echo -e '[fftw]\nlibraries = fftw3\n[fftw_mpi]\nlibraries = fftw3_mpi' > setup.cfg
pip install -e .
cd ..

# create temp folder
mkdir temp
