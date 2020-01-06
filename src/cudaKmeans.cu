#include "imagesHandler.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <random>
#include <vector>
#include <stdexcept>
#include <string>


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

//1)Compute the best distances between the 3 dimensional points in data_x,y,z
//2)if is the last iteration store the best cluster of each point in the data_assignments vector as a float id (0,1...)
//3)compute the new clusters means
__global__ void assign_clusters(const float* __restrict__ data_x,
                                const float* __restrict__ data_y,
                                const float* __restrict__ data_z,
                                float*  data_assignments,
                                int data_size,
                                const float* __restrict__ means_x,
                                const float* __restrict__ means_y,
                                const float* __restrict__ means_z,
                                float* __restrict__ new_sums_x,
                                float* __restrict__ new_sums_y,
                                float* __restrict__ new_sums_z,
                                int numberOfCluster,
                                int* __restrict__ counts,
                                bool save) {

	//With M threads per block a unique index for each thread is given by:int index = threadIdx.x + blockIdx.x * M;
	//Where M is the size of the block of threads; i.e.,blockDim.x

	extern __shared__ float shared_means[];

	const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= data_size) return;

	//first k threads copy over the cluster means.
	if (threadIdx.x < numberOfCluster) {
	    shared_means[threadIdx.x] = means_x[threadIdx.x];
	    shared_means[numberOfCluster + threadIdx.x] = means_y[threadIdx.x];
	    shared_means[numberOfCluster*2 + threadIdx.x] = means_z[threadIdx.x];
	}

	// Wait for those k threads.
	__syncthreads();

	const float x = data_x[index];
	const float y = data_y[index];
	const float z = data_z[index];

	float best_distance = FLT_MAX;
	int best_cluster=0;
	for (int cluster = 0; cluster < numberOfCluster; ++cluster) {
	const float distance =squared_l2_distance(x, y, z, shared_means[cluster],shared_means[numberOfCluster + cluster],shared_means[numberOfCluster*2 + cluster]);
		if (distance < best_distance) {

			best_distance = distance;
			best_cluster = cluster;

			if (save){
				data_assignments[index]=best_cluster;
			}
		}
	}

	atomicAdd(&new_sums_x[best_cluster], x);
	atomicAdd(&new_sums_y[best_cluster], y);
	atomicAdd(&new_sums_z[best_cluster], z);
	atomicAdd(&counts[best_cluster], 1);

}


// Each thread is one cluster, which just recomputes its coordinates as the mean of all points
//  assigned to it simply assigning the new passed means and dividing for the clusters number element
__global__ void compute_new_means(float* __restrict__ means_x,
                                  float* __restrict__ means_y,
                                  float* __restrict__ means_z,
                                  const float* __restrict__ new_sum_x,
                                  const float* __restrict__ new_sum_y,
                                  const float* __restrict__ new_sum_z,
                                  const int* __restrict__ counts
                                  ) {
  const int cluster = threadIdx.x;
  // Threshold count to turn 0/0 into 0/1.
  const int count = max(1, counts[cluster]);
  means_x[cluster] = new_sum_x[cluster] / count;
  means_y[cluster] = new_sum_y[cluster] / count;
  means_z[cluster] = new_sum_z[cluster] / count;
}



int main(int argc, char **argi)
{

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
    std::vector<float> assignments(rows * columns);

	//Data array population
	handler.dataAcquisition(h_x, h_y, h_z);
    Data d_data(h_x.size(), h_x, h_y, h_z,assignments);

    //Random first cluster means selections
    std::random_device seed;
    std::mt19937 rng(seed());
    std::shuffle(h_x.begin(), h_x.end(), rng);
    std::shuffle(h_y.begin(), h_y.end(), rng);
    std::shuffle(h_z.begin(), h_z.end(), rng);

	Data d_means(numberOfClusters, h_x, h_y, h_z, assignments);
	Data d_sums(numberOfClusters);

	//GPU initialization

	int* d_counts;
	cudaMalloc(&d_counts, numberOfClusters * sizeof(int));
	cudaMemset(d_counts, 0, numberOfClusters * sizeof(int));
	int number_of_elements = h_x.size();
	const int threads = 1024;
	const int blocks = (number_of_elements + threads - 1) / threads;
	const int shared_memory = d_means.bytes * 3;


	//boolean variable to saving assignments during the last iteration
	bool save = false;

	std::cout<< "\n\n image processing...\n\n";

	//clock initialization
	std::clock_t start;
	double duration;
	start = std::clock();

	//KMEANS
	for (size_t iteration = 0; iteration < iterations; ++iteration) {

		 cudaMemset(d_counts, 0, numberOfClusters * sizeof(int));
		 d_sums.clear();


		//last iteration saving
		if(iteration == iterations -1){
			save = true;
		}

		assign_clusters<<<blocks, threads, shared_memory>>>(d_data.x,
											 d_data.y,
											 d_data.z,
											 d_data.assignments,
											 d_data.size,
											 d_means.x,
											 d_means.y,
											 d_means.z,
											 d_sums.x,
											 d_sums.y,
											 d_sums.z,
											 numberOfClusters,
											 d_counts,
											 save);

		cudaDeviceSynchronize();

		compute_new_means<<<1, numberOfClusters>>>(d_means.x,
									d_means.y,
									d_means.z,
									d_sums.x,
									d_sums.y,
									d_sums.z,
									d_counts
									);

		cudaDeviceSynchronize();

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

