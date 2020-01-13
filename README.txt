C++ implementation
--------------------------------

compiler settings:
g++ -L/usr/X11R6/lib -lm -lpthread -lX11 ./src/imagesHandler.cpp ./src/ckmeans.cpp -o ckmeans

usage example:
./ckmeans ./testImages/car.jpg RGB 5 100 [display]



OMP implementation 
-----------------------------------

compiler settings:
g++ -L/usr/X11R6/lib -lm -lpthread -lX11 -fopenmp ./src/imagesHandler.cpp ./src/ompKmeans.cpp -o ompkmeans  

usage example:
./ompkmeans ./testImages/car.jpg RGB 5 100 [display]



CUDA implementation
--------------------------------

compiler settings:
nvcc -L/usr/X11R6/lib -lm -lpthread -lX11 ./src/imagesHandler.cpp ./src/cudaKmeans.cu -o cudaKmeans
nvcc -L/usr/X11R6/lib -lm -lpthread -lX11 ./src/imagesHandler.cpp ./src/fastCudaKmeans.cu -o fastCudaKmeans
nvcc -L/usr/X11R6/lib -lm -lpthread -lX11 ./src/imagesHandler.cpp ./src/reductionWithoutClusterLimitation.cu -o modified

usage example:
./cudaKmeans ./testImages/car.jpg RGB 5 100 [display]
./fastCudaKmeans ./testImages/car.jpg RGB 5 100 [display]


