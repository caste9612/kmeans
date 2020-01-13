#include "imagesHandler.h"
#include <omp.h>
#include <sys/time.h>


struct Point{
  float x{0}, y{0}, z{0};
};

using DataFrame = std::vector<Point>;

//KMEANS
float square(float value) {
  return value * value;
}

float squared_l2_distance(Point first, Point second) {
  return (square(first.x - second.x) + square(first.y - second.y) + square(first.z - second.z));
}

std::vector<int> k_means(const DataFrame& data,int k,int number_of_iterations, std::vector<int>& clustColorR, std::vector<int>& clustColorG, std::vector<int>& clustColorB) {

  //Centroids init
  static std::random_device seed;
  std::mt19937 rng(seed());
  std::uniform_int_distribution<int> indices(0, data.size() - 1);
 
  DataFrame means(k);
  //Centroids population
  for (int i=0;i<k;i++) {
    means[i]=data[indices(rng)];
  }

  //init assignments vector
  std::vector<int> assignments(data.size());

  //For each iteration (user input)
  for (int iteration = 0; iteration < number_of_iterations; ++iteration) {
    //Find assignemnts
  #pragma omp parallel for 
    for (int point = 0; point < data.size(); point++) {
      float best_distance = FLT_MAX;
      int best_cluster = 0;
      for (int cluster = 0; cluster < k; ++cluster) {
        const float distance = squared_l2_distance(data[point], means[cluster]);
        if (distance < best_distance) {
          best_distance = distance;
          best_cluster = cluster;
        }
      }
      assignments[point] = best_cluster;
    }

    //Sum values and count points for each cluster
    DataFrame new_means(k);
    std::vector<int> counts(k, 0);

    for (int point = 0; point < data.size(); ++point) {
      const auto cluster = assignments[point];
      new_means[cluster].x += data[point].x;
      new_means[cluster].y += data[point].y;
      new_means[cluster].z += data[point].z;
      counts[cluster] += 1;
    }

    //Divide by the count found to calculate new means
	#pragma omp parallel for
    for (int cluster = 0; cluster < k; cluster++) {
      // Turn 0/0 into 0/1 to avoid zero division.
      const auto count = std::max<int>(1, counts[cluster]);
      means[cluster].x = new_means[cluster].x / count;
      means[cluster].y = new_means[cluster].y / count;
      means[cluster].z = new_means[cluster].z/ count;
    }
  }
  
  	#pragma omp parallel for
	for (int cluster = 0; cluster < k; cluster++){

		clustColorR[cluster]=(int)means[cluster].x;
		clustColorG[cluster]=(int)means[cluster].y;
		clustColorB[cluster]=(int)means[cluster].z;
	}

  return assignments;
}


//MAIN
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
	DataFrame data(rows*columns);

	handler.dataAcquisition(h_x, h_y, h_z);
	for(int i=0;i<columns;i++){
		for(int j=0;j<rows;j++){
			Point point;
			point.x=h_x[i * rows + j];
			point.y=h_y[i * rows + j];
			point.z=h_z[i * rows + j];
			data[i * rows + j]=point;
		}
	}
	
	//KMEANS

	std::vector<int> assignedPixels;
	std::cout<< "\n\n image processing...\n\n";

	//clock initialization (openmp ad hoc)
	struct timeval start, end;
	gettimeofday(&start, NULL);


	//Data processing
	std::vector<int> clustColorR(numberOfClusters);
	std::vector<int> clustColorG(numberOfClusters);
	std::vector<int> clustColorB(numberOfClusters);

	assignedPixels = k_means(data, numberOfClusters, iterations, clustColorR, clustColorG, clustColorB);


  	gettimeofday(&end, NULL);

  	long delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
          end.tv_usec - start.tv_usec) / 1.e6;

  	std::cout<< "PROCESSING TIME: "<< delta << " s" <<'\n';

	int* ap = &assignedPixels[0];

  	handler.disp(ap, clustColorR, clustColorG, clustColorB);
}
