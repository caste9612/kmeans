#include "imagesHandler.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

    cudaMemcpy(x, h_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_y.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(z, h_z.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(assignments, h_assignments.data(), bytes, cudaMemcpyHostToDevice);
  }

  Data(int size, float* h_x, float* h_y, float* h_z, float* h_assignments): size(size),bytes(size*sizeof(float)){
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMalloc(&z, bytes);
    cudaMalloc(&assignments, bytes);

    cudaMemcpy(x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_y, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(z, h_z, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(assignments, h_assignments, bytes, cudaMemcpyHostToDevice);;
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


__global__ void assign_clusters(const float* __restrict__ data_x,
  const float* __restrict__ data_y,
  const float* __restrict__ data_z,
  float*  data_assignments,
  int data_size,
  const float* __restrict__ means_x,
  const float* __restrict__ means_y,
  const float* __restrict__ means_z,
  const  int numberOfCluster) {


    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= data_size) return;


  const float x = data_x[index];
  const float y = data_y[index];
  const float z = data_z[index];


  float best_distance = FLT_MAX;
  int best_cluster=0;
  for (int cluster = 1; cluster < numberOfCluster; cluster++) {
    const float distance =squared_l2_distance(x, y, z, means_x[cluster],means_y[cluster],means_z[cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

data_assignments[index]=best_cluster;

}


__global__ void assign_clustersModified(const float* __restrict__ data_x,
  const float* __restrict__ data_y,
  const float* __restrict__ data_z,
  const int  data_size,
  float*  means_x,
  float*  means_y,
  float*  means_z,
  float* means_assignments,
  const int  numberOfClusters) {


    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= data_size) return;


    //const float x = data_x[index];
    //const float y = data_y[index];
    //const float z = data_z[index];


  float best_distance = FLT_MAX;
  int best_cluster=0;
  for (int cluster = 0; cluster < numberOfClusters; cluster++) {
    const float distance =squared_l2_distance(data_x[index], data_y[index], data_z[index], means_x[cluster],means_y[cluster],means_z[cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  for(int cluster = 0; cluster < numberOfClusters; cluster++){
    if(cluster == best_cluster){
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

__global__ void populate( float*  data_x,
                          float*  data_y,
                          float*  data_z,
                          float*  data_assignments,
                          float*  copy_x,
                          float*  copy_y,
                          float*  copy_z,
                          float*  copy_assignments,
                          int data_size,
                          int numberOfCluster,
                          float*  counts,
                          const int cluster) {

  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_index >= data_size) return;

  if (data_assignments[global_index] != cluster){
    copy_x[global_index] = 0;
    copy_y[global_index] = 0;
    copy_z[global_index] = 0;
    counts[global_index] = 0;
  }else{
    copy_x[global_index] = data_x[global_index];
    copy_y[global_index] = data_y[global_index];
    copy_z[global_index] = data_z[global_index];
    counts[global_index]=1;
  }

}

/*
__device__ void populateSingle( float*  data_x,
  float*  data_assignments,
  float*  copy_x,
  int data_size,
  const int cluster,
  int mode) {

  unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_index >= data_size) return;

  if (data_assignments[global_index] != cluster){
    copy_x[global_index] = 0;
  }else{
    if(mode == 0 ){copy_x[global_index] = data_x[global_index];}
    else{counts[global_index]=1;}
  }
}
*/

template <size_t blockSize>
__device__ void warpReduce(volatile float *sdata, size_t tid)
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
    if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
    if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
    if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

/*

template <size_t blockSize>
__global__ void reduceCUDA(float* g_idata, float* g_odata, size_t n)
{
    __shared__ float sdata[blockSize];

    size_t tid = threadIdx.x;
    size_t i = blockIdx.x*(blockSize) + tid;
    size_t gridSize = blockSize*gridDim.x;
    sdata[tid] = 0;

    while (i < n) { sdata[tid] += g_idata[i]; i += gridSize; }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >=  512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >=  256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >=  128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}





template<size_t blockSize>
float GPUReduction(float* dA, size_t N)
{
    float tot = 0.;
    size_t n = N;
    size_t blocksPerGrid = std::ceil((1.*n) / blockSize);

    float* tmp;
    cudaMalloc(&tmp, sizeof(float) * blocksPerGrid); checkCUDAError("Error allocating tmp [GPUReduction]");

    float* from = dA;

    do
    {
        blocksPerGrid   = std::ceil((1.*n) / blockSize);
        reduceCUDA<blockSize><<<blocksPerGrid, blockSize>>>(from, tmp, n);
        from = tmp;
        n = blocksPerGrid;
    } while (n > blockSize);

    if (n > 1){
        reduceCUDA<blockSize><<<1, blockSize>>>(tmp, tmp, n);
    }

    cudaDeviceSynchronize();
    checkCUDAError("Error launching kernel [GPUReduction]");

    cudaMemcpy(&tot, tmp, sizeof(float), cudaMemcpyDeviceToHost); checkCUDAError("Error copying result [GPUReduction]");
    cudaFree(tmp);
    return tot;
}

*/

template<size_t blockSize>
__global__ void reductionModified( float*  data_x,
  float*  data_y,
  float*  data_z,
  float*  data_assignments,
  float*  dataCopy_x,
  float* dataCopy_y,
  float* dataCopy_z,
  float* dataCopy_assignments,
  size_t data_size,
  int numberOfClusters) {

    size_t tid = threadIdx.x;
    size_t i = blockIdx.x*(blockSize) + tid;
    size_t gridSize = blockSize*gridDim.x;

    __shared__  float sdata [BLOCKSIZE*4] ;

    for(int cluster = 0; cluster<numberOfClusters; cluster++){

      int x = 0;
      int y = data_size;
      int z = data_size * 2;
      int ce = data_size * 3;

      if(i < data_size){
      sdata[i + x] = data_x[i + data_size * cluster ];
      sdata[i + y] = data_y[i + data_size * cluster ];
      sdata[i + z] = data_z[i + data_size * cluster ];
      sdata[i + ce]= data_assignments[i + data_size * cluster ];
    }

    __syncthreads();

    size_t t = i + gridSize;
    while (t < data_size) { 
      sdata[tid + x] += sdata[t + x]; 
      sdata[tid + y] += sdata[t + y]; 
      sdata[tid + z] += sdata[t +z]; 
      sdata[tid + ce] += sdata[t + ce]; 

      t += gridSize; 
    }
    __syncthreads();

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
      //const int cluster_index = blockIdx.x * numberOfClusters + cluster;
      dataCopy_x[blockIdx.x + gridDim.x * cluster] = sdata[x];
      dataCopy_y[blockIdx.x + gridDim.x * cluster] = sdata[y];
      dataCopy_z[blockIdx.x + gridDim.x * cluster] = sdata[z];
      dataCopy_assignments[blockIdx.x + gridDim.x * cluster] = sdata[ce];
    } 
  }
}

__global__ void divideStep( float*  dmx,
  float*  dmy,
  float*  dmz,
  float*  dma,
  float*  tmpx,
  float* tmpy,
  float* tmpz,
  float* tmpa,
  size_t data_size,
  int numberOfClusters) {

    int count = (int)tmpa[threadIdx.x];
    if(count < 1){
      count = 1;
    }


    dmx[threadIdx.x] /= count;
    dmy[threadIdx.x] /= count;
    dmz[threadIdx.x] /= count;

  }
/*




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

  Data d_data(h_x.size(), h_x, h_y, h_z,h_assignments);checkCUDAError("Error during d_data init");
  Data copy(h_x.size(), d_data.x, d_data.y, d_data.z, d_data.assignments);checkCUDAError("Error during copy init");


  //Random first cluster means selections
  std::random_device seed;
  std::mt19937 rng(seed());
  std::shuffle(h_x.begin(), h_x.end(), rng);
  std::shuffle(h_y.begin(), h_y.end(), rng);
  std::shuffle(h_z.begin(), h_z.end(), rng);

  Data d_means(numberOfClusters, h_x, h_y, h_z, h_assignments);checkCUDAError("Error during d_means init");

  int number_of_elements = h_x.size();
  const int threads = 1024;
  const int blocks = (number_of_elements + threads + 1) / (threads);

	//GPU initialization

	float* d_counts;
  cudaMalloc(&d_counts, number_of_elements  * sizeof(float));checkCUDAError("Error during counts init");  

  std::cout<< "\n\n image processing...\n\n";

	//clock initialization
	std::clock_t start;
	double duration;
  start = std::clock();

	//KMEANS
	for (int iteration = 0; iteration < iterations; iteration++) {

    assign_clusters<<<blocks, threads>>>(d_data.x,
      d_data.y,
      d_data.z,
      d_data.assignments,
      h_x.size(),
      d_means.x,
      d_means.y,
      d_means.z,
      numberOfClusters);checkCUDAError("Error during assign cluster ");

    cudaDeviceSynchronize();

    std::vector<float> d_meansx(rows * columns);
    std::vector<float> d_meansy(rows * columns);
    std::vector<float> d_meansz(rows * columns);
 
    for (int cluster = 0; cluster < numberOfClusters; cluster++) {
     

      cudaMemset(d_counts, 0, number_of_elements  * sizeof(float));checkCUDAError("Error count set ");

      //TODO: eliminare counts  con copy assignments
      populate<<<blocks, threads>>>(d_data.x,
        d_data.y,
        d_data.z,
        d_data.assignments,
        copy.x,
        copy.y,
        copy.z,
        copy.assignments,
        number_of_elements,
        numberOfClusters,
        d_counts,
        cluster);checkCUDAError("Error during population");
        

      cudaDeviceSynchronize();

      float tmp = GPUReduction<BLOCKSIZE>(copy.x,h_x.size());
      d_meansx[cluster] = tmp;
      tmp = GPUReduction<BLOCKSIZE>(copy.y,h_x.size());
      d_meansy[cluster] = tmp;
      tmp = GPUReduction<BLOCKSIZE>(copy.z,h_x.size());
      d_meansz[cluster] = tmp;

      float clusterElements = GPUReduction<BLOCKSIZE>(d_counts, number_of_elements);

      const int count = std::max(1, (int)clusterElements);

      d_meansx[cluster] /= count;
      d_meansy[cluster] /= count;
      d_meansz[cluster] /= count;

    }
    cudaMemcpy(d_means.x, d_meansx.data(), numberOfClusters*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_means.y, d_meansy.data(), numberOfClusters*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_means.z, d_meansz.data(), numberOfClusters*sizeof(float), cudaMemcpyHostToDevice);
  }

  duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  std::cout<< "PROCESSING TIME: "<< duration << " s" <<'\n';

  //Processed data acquisition to coloring output image
  float* h_best;
  h_best = (float*)malloc(number_of_elements*sizeof(float));
  cudaMemcpy(h_best,d_data.assignments, number_of_elements*sizeof(float), cudaMemcpyDeviceToHost);

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
  std::cout<<(int)finalmeanx[cluster]<<" "<<(int)finalmeany[cluster]<<" "<<(int)finalmeanz[cluster]<<" "<<std::endl;
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
*/





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

  Data d_data(h_x.size(), h_x, h_y, h_z,h_assignments);checkCUDAError("Error during d_data init");
  Data copy(h_x.size(), d_data.x, d_data.y, d_data.z, d_data.assignments);checkCUDAError("Error during copy init");


  //Random first cluster means selections
  std::random_device seed;
  std::mt19937 rng(seed());
  std::shuffle(h_x.begin(), h_x.end(), rng);
  std::shuffle(h_y.begin(), h_y.end(), rng);
  std::shuffle(h_z.begin(), h_z.end(), rng);

  int number_of_elements = h_x.size();
 
  Data d_means(numberOfClusters * number_of_elements, h_x, h_y, h_z, h_assignments);checkCUDAError("Error during d_means init");

 

  //GPU initialization
  //const int threads = 1024;
  //const int blocks = (number_of_elements + threads + 1) / (threads);

  std::cout<< "\n\n image processing...\n\n";

	//clock initialization
	std::clock_t start;
	double duration;
  start = std::clock();

  size_t blocksPerGrid = std::ceil((1.*number_of_elements) / BLOCKSIZE);


	//KMEANS
	for (int iteration = 0; iteration < iterations; iteration++) {

    assign_clustersModified<<<blocksPerGrid, BLOCKSIZE>>>(d_data.x,
      d_data.y,
      d_data.z,
      number_of_elements,
      d_means.x,
      d_means.y,
      d_means.z,
      d_means.assignments,
      numberOfClusters);checkCUDAError("Error during assign cluster ");

    cudaDeviceSynchronize();

    //riduci
    size_t n = number_of_elements;

    float* tmpx;
    cudaMalloc(&tmpx, sizeof(float) * blocksPerGrid * numberOfClusters); checkCUDAError("Error allocating tmp [GPUReduction]");
    float* tmpy;
    cudaMalloc(&tmpy, sizeof(float) * blocksPerGrid * numberOfClusters); checkCUDAError("Error allocating tmp [GPUReduction]");
    float* tmpz;
    cudaMalloc(&tmpz, sizeof(float) * blocksPerGrid * numberOfClusters); checkCUDAError("Error allocating tmp [GPUReduction]");
    float* tmpass;
    cudaMalloc(&tmpass, sizeof(float) * blocksPerGrid * numberOfClusters); checkCUDAError("Error allocating tmp [GPUReduction]");

      do{
        blocksPerGrid   = std::ceil((1.*n) / BLOCKSIZE);
  
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

            //sostituire con funzione kernel che copia
            cudaMemcpy(d_means.x, tmpx, sizeof(float) * blocksPerGrid * numberOfClusters, cudaMemcpyDeviceToDevice);checkCUDAError("Error copying into tmp");
            cudaMemcpy(d_means.y, tmpy, sizeof(float) * blocksPerGrid * numberOfClusters,cudaMemcpyDeviceToDevice);checkCUDAError("Error copying into tmp");
            cudaMemcpy(d_means.z, tmpz, sizeof(float) * blocksPerGrid * numberOfClusters,cudaMemcpyDeviceToDevice);checkCUDAError("Error copying into tmp");
            cudaMemcpy(d_means.assignments, tmpass, sizeof(float) * blocksPerGrid * numberOfClusters,cudaMemcpyDeviceToDevice);checkCUDAError("Error copying into tmp");

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

            //funzione kernel che copia dentro d_means.x,y,z il contenuto di tmpx,y,z
            //diviso il cout= std::max(1,tmpass[cluster])

            divideStep<<<1,numberOfClusters>>>(d_means.x,
              d_means.y,
              d_means.z,
              d_means.assignments,
              tmpx,
              tmpy,
              tmpz,
              tmpass,
              n,
              numberOfClusters);checkCUDAError("Error during divideStep");
        }

        cudaDeviceSynchronize();
        checkCUDAError("Error launching kernel [GPUReduction]");
    
        cudaFree(tmpx);
        cudaFree(tmpy);
        cudaFree(tmpz);
        cudaFree(tmpass);

    }
  

  duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  std::cout<< "PROCESSING TIME: "<< duration << " s" <<'\n';

  //Processed data acquisition to coloring output image
  float* h_best;
  h_best = (float*)malloc(number_of_elements*sizeof(float));
  cudaMemcpy(h_best,d_data.assignments, number_of_elements*sizeof(float), cudaMemcpyDeviceToHost);

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
  std::cout<<(int)finalmeanx[cluster]<<" "<<(int)finalmeany[cluster]<<" "<<(int)finalmeanz[cluster]<<" "<<std::endl;
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

