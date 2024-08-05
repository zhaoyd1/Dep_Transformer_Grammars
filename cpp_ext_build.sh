mkdir .dependencies
cd .dependencies
git clone --depth 1 -b 20220623.1 https://github.com/abseil/abseil-cpp.git
git clone --depth 1 -b 3.4.0 https://gitlab.com/libeigen/eigen.git
git clone --depth 1 -b v2.10.2 https://github.com/pybind/pybind11.git

cd ../masking_bllip
rm -rf tmp
mkdir tmp
cd tmp
cmake ..
cmake --build . -- -j
mv ./*.so ..
cd ../..
