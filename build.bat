md pplcv-build
cd pplcv-build
cmake %* -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install ..
cmake --build . --config Release
cmake --install . --config Release
cd ..
