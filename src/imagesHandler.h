/*
 * imagesHandler.h
 *
 *  Created on: 31/ott/2019
 *      Author: Antonio&Michela
 */
 
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <vector>
#include <stdio.h>
#include <cassert>
#include <limits>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <cfloat>



#ifndef IMAGESHANDLER_H_
#define IMAGESHANDLER_H_

class imagesHandler {

private:

	int numberOfClusters;
	int iterations;
	int columns;
	int rows;
	std::string fileName;
	std::string mode;
	std::string dispImages;



public:

	imagesHandler();
	virtual ~imagesHandler();

	std::vector<int> inputParamAcquisition(char **argi);

	void dataAcquisition(std::vector<float> &h_x,std::vector<float> &h_y,std::vector<float> &h_z);

	void disp(int* assignedPixels, std::vector<int> clusterColorR, std::vector<int> clusterColorG, std::vector<int> clusterColorB);
};

#endif /* IMAGESHANDLER_H_ */
