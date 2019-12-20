C++ implementation
--------------------------------

compiler settings:
g++ -L/usr/X11R6/lib -lm -lpthread -lX11 ./src/imagesHandler.cpp ./src/ckmeans.cpp -o ckmeans

usage example:
./ckmeans ./testImages/car.jpg RGB 5 100

OMP implementation 
-----------------------------------






CUDA implementation
--------------------------------

compiler settings:
nvcc -L/usr/X11R6/lib -lm -lpthread -lX11 ./src/imagesHandler.cpp ./src/cudaKmeans.cu -o cudakmeans


usage example:
./cudakmeans ./testImages/car.jpg RGB 5 100

