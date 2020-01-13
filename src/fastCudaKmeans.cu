#include "imagesHandler.h"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <string>
#define BLOCKSIZE 1024

void checkCUDAError(const char *msg){
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err){
        fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(EXIT_FAILURE); 
    }
}

struct Data {
  explicit Data(int size) : size(size), bytes(size * sizeof(float)){
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMalloc(&z, bytes);
    cudaMalloc(&assignments, bytes);
  }

  Data(int size, std::vector<float>& h_x, std::vector<float>& h_y,std::vector<float>& h_z,std::vector<float>& h_assignments): size(size),bytes(size*sizeof(float)){
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMalloc(&z, bytes);
    cudaMalloc(&assignments, bytes);

    cudaMemcpy(x, h_x.data(), h_x.size()* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_y.data(), h_x.size()* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(z, h_z.data(), h_x.size()* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(assignments, h_assignments.data(), h_x.size()* sizeof(float), cudaMemcpyHostToDevice);
  }

  ~Data() {
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(assignments);
  }

  void clear() {
    cudaMemset(x, 0, bytes);
    cudaMemset(y, 0, bytes);
    cudaMemset(z, 0, bytes);
    cudaMemset(assignments, 0, bytes);
  }

  float* x{nullptr};
  float* y{nullptr};
  float* z{nullptr};
  float* assignments{nullptr};
  
  int size{0};
  int bytes{0};

};

//function to easily compute l2 distance, can be quickly updated with more dimensions adding parameters
__device__ float squared_l2_distance(float x_1, float y_1, float z_1, float x_2, float y_2, float z_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2) + (z_1 - z_2) * (z_1 - z_2);
}

//function to compute the distances AND write the cluster id into data_assignment
__global__ void assign_clusters(const float* __restrict__ data_x,
                                const float* __restrict__ data_y,
                                const float* __restrict__ data_z,
                                float* data_assignments,
                                const int  data_size,
                                const float* __restrict__  means_x,
                                const float* __restrict__  means_y,
                                const float* __restrict__  means_z,
                                const float* __restrict__ means_assignments,
                                const int  numberOfClusters) {
  
	__shared__ float shared_means[300*3];

	const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= data_size) return;

	//first k threads copy over the cluster means.
	if (threadIdx.x < numberOfClusters) {
		shared_means[threadIdx.x] = means_x[threadIdx.x];
		shared_means[numberOfClusters + threadIdx.x] = means_y[threadIdx.x];
		shared_means[numberOfClusters*2 + threadIdx.x] = means_z[threadIdx.x];
	}

	// Wait for those k threads.
	__syncthreads();

	const float x = data_x[index];
	const float y = data_y[index];
	const float z = data_z[index];

	float best_distance = squared_l2_distance(x, y, z,shared_means[0],shared_means[numberOfClusters],shared_means[numberOfClusters*2]);
	int best_cluster = 0;
	for (int cluster = 1; cluster < numberOfClusters; cluster++) {
    	float distance =squared_l2_distance(x, y, z, shared_means[cluster],shared_means[numberOfClusters + cluster],shared_means[numberOfClusters*2 + cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }
  data_assignments[index]=best_cluster;
}

//populate the big 4 array for reductions
__global__ void populate(const float* __restrict__ data_x,
                        const float* __restrict__ data_y,
                        const float* __restrict__ data_z,
                        const float* __restrict__ data_assignments,
                        const int  data_size,
                        float*  means_x,
                        float*  means_y,
                        float*  means_z,
                        float* means_assignments,
                        const int  numberOfClusters) {

  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= data_size) return;

  for(int cluster = 0; cluster < numberOfClusters; cluster++){
    if(cluster == data_assignments[index]){
      means_x[index + data_size * cluster] = data_x[index];
      means_y[index + data_size * cluster] = data_y[index];
      means_z[index + data_size * cluster] = data_z[index];
      means_assignments[index + data_size * cluster] = 1;
    }else{
      means_x[index + data_size * cluster] = 0;
      means_y[index + data_size * cluster] = 0;
      means_z[index + data_size * cluster] = 0;
      means_assignments[index + data_size * cluster] = 0;
    }
  }
}

//REDUCTION

template <size_t blockSize>
__device__ void warpReduce(volatile float *sdata, size_t tid){
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
    if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
    if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
    if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}


template<size_t blockSize>
__global__ void reductionModified(const float* __restrict__  data_x,
                                  const float* __restrict__  data_y,
                                  const float* __restrict__  data_z,
                                  const float* __restrict__  data_assignments,
                                  float*  dataOutput_x,
                                  float* dataOutput_y,
                                  float* dataOutput_z,
                                  float* dataOutput_assignments,
                                  size_t data_size,
                                  int numberOfClusters) {

    __shared__  float sdata [blockSize*4] ;

    for(int cluster = 0; cluster<numberOfClusters; cluster++){
      size_t tid = threadIdx.x;
      size_t i = blockIdx.x*(blockSize * 2) + tid;
      size_t gridSize = blockSize * 2 *gridDim.x;

      int x = 0;
      int y = blockSize;
      int z = blockSize * 2;
      int ce = blockSize * 3;

      sdata[tid + x] =  0;
      sdata[tid + y] = 0;
      sdata[tid + z] = 0;
      sdata[tid + ce] = 0;

      while(i < data_size){
        sdata[tid + x] += data_x[i + data_size * cluster ] + data_x[i + data_size * cluster +blockSize];
        sdata[tid + y] += data_y[i + data_size * cluster ]+ data_y[i + data_size * cluster +blockSize];
        sdata[tid + z] += data_z[i + data_size * cluster ]+ data_z[i + data_size * cluster +blockSize];
        sdata[tid + ce] += data_assignments[i + data_size * cluster ] + data_assignments[i + data_size * cluster +blockSize];
        i += gridSize;
    } __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) {
      sdata[tid + x] += sdata[tid + x + 512]; 
      sdata[tid + y] += sdata[tid + y + 512]; 
      sdata[tid + z] += sdata[tid + z + 512]; 
      sdata[tid + ce] += sdata[tid + ce + 512]; 
    } __syncthreads(); }

    if (blockSize >=  512) { if (tid < 256) { 
      sdata[tid + x] += sdata[tid + x + 256]; 
      sdata[tid + y] += sdata[tid + y + 256]; 
      sdata[tid + z] += sdata[tid + z + 256]; 
      sdata[tid + ce] += sdata[tid + ce + 256];
    } __syncthreads(); }

    if (blockSize >=  256) { if (tid < 128) { 
      sdata[tid + x] += sdata[tid + x + 128]; 
      sdata[tid + y] += sdata[tid + y + 128]; 
      sdata[tid + z] += sdata[tid + z + 128]; 
      sdata[tid + ce] += sdata[tid + ce + 128];   
    } __syncthreads(); }

    if (blockSize >=  128) { if (tid <  64) { 
      sdata[tid + x] += sdata[tid + x + 64]; 
      sdata[tid + y] += sdata[tid + y + 64]; 
      sdata[tid + z] += sdata[tid + z + 64]; 
      sdata[tid + ce] += sdata[tid + ce + 64];      
    } __syncthreads(); }

    if (tid < 32){
      warpReduce<blockSize>(sdata, tid + x);
      warpReduce<blockSize>(sdata, tid + y);
      warpReduce<blockSize>(sdata, tid + z);
      warpReduce<blockSize>(sdata, tid + ce);
    } __syncthreads();

    if (tid == 0){
      dataOutput_x[blockIdx.x + gridDim.x * cluster] = sdata[x];
      dataOutput_y[blockIdx.x + gridDim.x * cluster] = sdata[y];
      dataOutput_z[blockIdx.x + gridDim.x * cluster] = sdata[z];
      dataOutput_assignments[blockIdx.x + gridDim.x * cluster] = sdata[ce];
    } 
  }
}

//fuction to compute new means with reduction results
__global__ void divideStep(float*  dmx,
                          float*  dmy,
                          float*  dmz,
                          const float* __restrict__  tmpx,
                          const float* __restrict__ tmpy,
                          const float* __restrict__ tmpz,
                          const float* __restrict__ tmpa,
                          int numberOfClusters) {

    if(threadIdx.x >= numberOfClusters)return;

    int count = max(1,(int)tmpa[threadIdx.x]);

    dmx[threadIdx.x] = tmpx[threadIdx.x]/count;
    dmy[threadIdx.x] = tmpy[threadIdx.x]/count;
    dmz[threadIdx.x] = tmpz[threadIdx.x]/count;
}




int main(int argc, char **argi){

	//Image Handler creation
	imagesHandler handler;

	//Input params acqisition && Image opening by CImg and dimension acquisition
	std::vector<int> params = handler.inputParamAcquisition(argi);
	
	int iterations = params[0];
	int numberOfClusters = params[1];
	int columns = params[2];
	int rows = params[3];
  
	//Data array initialization
	std::vector<float> h_x(rows * columns);
	std::vector<float> h_y(rows * columns);
	std::vector<float> h_z(rows * columns);
	std::vector<float> h_assignments(rows * columns);
	for(int i=0;i<rows*columns;i++){
		h_assignments[i]=0;
	}

	//Data array population   
	handler.dataAcquisition(h_x, h_y, h_z);

	int number_of_elements = h_x.size();

	Data d_data(number_of_elements, h_x, h_y, h_z,h_assignments);checkCUDAError("Error during d_data init");

	//Random first cluster means selections
	std::random_device seed;
	std::mt19937 rng(seed());
	std::shuffle(h_x.begin(), h_x.end(), rng);
	std::shuffle(h_y.begin(), h_y.end(), rng);
	std::shuffle(h_z.begin(), h_z.end(), rng);

	Data d_means(numberOfClusters * number_of_elements, h_x, h_y, h_z, h_assignments);checkCUDAError("Error during d_means init");

	//GPU initialization
	size_t blocksPerGridFixed = std::ceil((1.*number_of_elements) / BLOCKSIZE);

	float* tmpx;
	cudaMalloc(&tmpx, sizeof(float) * blocksPerGridFixed * numberOfClusters); checkCUDAError("Error allocating tmp [GPUReduction]");
	float* tmpy;
	cudaMalloc(&tmpy, sizeof(float) * blocksPerGridFixed * numberOfClusters); checkCUDAError("Error allocating tmp [GPUReduction]");
	float* tmpz;
	cudaMalloc(&tmpz, sizeof(float) * blocksPerGridFixed * numberOfClusters); checkCUDAError("Error allocating tmp [GPUReduction]");
	float* tmpass;
	cudaMalloc(&tmpass, sizeof(float) * blocksPerGridFixed * numberOfClusters); checkCUDAError("Error allocating tmp [GPUReduction]");

	std::cout<< "\n\n image processing...\n\n";

	//clock initialization
	std::clock_t start;
	double duration;
  	start = std::clock();

	//KMEANS
	for (int iteration = 0; iteration < iterations; iteration++) {

    assign_clusters<<<blocksPerGridFixed, BLOCKSIZE>>>(d_data.x,
                                                      d_data.y,
                                                      d_data.z,
                                                      d_data.assignments,
                                                      number_of_elements,
                                                      d_means.x,
                                                      d_means.y,
                                                      d_means.z,
                                                      d_means.assignments,
                                                      numberOfClusters
                                                    );checkCUDAError("Error during assign cluster ");

    cudaDeviceSynchronize();

    populate<<<blocksPerGridFixed, BLOCKSIZE>>>(d_data.x,
                                    d_data.y,
                                    d_data.z,
                                    d_data.assignments,
                                    number_of_elements,
                                    d_means.x,
                                    d_means.y,
                                    d_means.z,
                                    d_means.assignments,
                                    numberOfClusters
                                  );checkCUDAError("Error during population");

    cudaDeviceSynchronize();

    //reduction
    size_t n = number_of_elements;

    do{
      size_t blocksPerGrid   = std::ceil((1.*n) / BLOCKSIZE);

      reductionModified<BLOCKSIZE><<<blocksPerGrid,BLOCKSIZE>>>(d_means.x,
                                                                d_means.y,
                                                                d_means.z,
                                                                d_means.assignments,
                                                                tmpx,
                                                                tmpy,
                                                                tmpz,
                                                                tmpass,
                                                                n,
                                                                numberOfClusters);checkCUDAError("Error during reduction");
          
        cudaDeviceSynchronize();checkCUDAError("Error on do-while loop [GPUReduction]");

        cudaMemcpy(d_means.x, tmpx, sizeof(float) * blocksPerGrid * numberOfClusters, cudaMemcpyDeviceToDevice);checkCUDAError("Error copying into tmpx");
        cudaMemcpy(d_means.y, tmpy, sizeof(float) * blocksPerGrid * numberOfClusters,cudaMemcpyDeviceToDevice);checkCUDAError("Error copying into tmpy");
        cudaMemcpy(d_means.z, tmpz, sizeof(float) * blocksPerGrid * numberOfClusters,cudaMemcpyDeviceToDevice);checkCUDAError("Error copying into tmpz");
        cudaMemcpy(d_means.assignments, tmpass, sizeof(float) * blocksPerGrid * numberOfClusters,cudaMemcpyDeviceToDevice);checkCUDAError("Error copying into tmpass");

        n = blocksPerGrid;

	} while (n > BLOCKSIZE);

    if (n > 1){
      reductionModified<BLOCKSIZE><<<1,BLOCKSIZE>>>(tmpx,
                                                    tmpy,
                                                    tmpz,
                                                    tmpass,
                                                    tmpx,
                                                    tmpy,
                                                    tmpz,
                                                    tmpass,
                                                    n,
                                                    numberOfClusters);checkCUDAError("Error during last step reduction");

      cudaDeviceSynchronize();checkCUDAError("Error on mid main loop [GPUReduction]");

      divideStep<<<1,BLOCKSIZE>>>(d_means.x,
                                  d_means.y,
                                  d_means.z,
                                  tmpx,
                                  tmpy,
                                  tmpz,
                                  tmpass,
                                  numberOfClusters);checkCUDAError("Error during divideStep");
      }
      cudaDeviceSynchronize();checkCUDAError("Error on bottom main loop [GPUReduction]");
    }
    cudaFree(tmpx);
    cudaFree(tmpy);
    cudaFree(tmpz);
    cudaFree(tmpass);
  
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<< "PROCESSING TIME: "<< duration << " s" <<'\n';

    //Processed data acquisition to coloring output image
    float* h_best;
    h_best = (float*)malloc(h_x.size()*sizeof(float));
    cudaMemcpy(h_best,d_data.assignments, h_x.size()*sizeof(float), cudaMemcpyDeviceToHost);

    float* finalmeanx;
    float* finalmeany;
    float* finalmeanz;
    finalmeanx = (float*)malloc(numberOfClusters*sizeof(float));
    finalmeany = (float*)malloc(numberOfClusters*sizeof(float));
    finalmeanz = (float*)malloc(numberOfClusters*sizeof(float));
    cudaMemcpy(finalmeanx, d_means.x, numberOfClusters*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(finalmeany, d_means.y, numberOfClusters*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(finalmeanz, d_means.z, numberOfClusters*sizeof(float),cudaMemcpyDeviceToHost);

    std::vector<int> clustColorR(numberOfClusters);
    std::vector<int> clustColorG(numberOfClusters);
    std::vector<int> clustColorB(numberOfClusters);

    for (int cluster = 0; cluster < numberOfClusters; cluster++){
      clustColorR[cluster]=(int)finalmeanx[cluster];
      clustColorG[cluster]=(int)finalmeany[cluster];
      clustColorB[cluster]=(int)finalmeanz[cluster];
    }

    int* assignedPixels;
    assignedPixels = (int*)malloc(number_of_elements*sizeof(int));
    for(int i=0; i<number_of_elements; i++){
      assignedPixels[i]=(int)h_best[i];
    }

    handler.disp(assignedPixels, clustColorR, clustColorG, clustColorB);
}
