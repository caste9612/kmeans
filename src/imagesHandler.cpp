/*
 * imagesHandler.cpp
 *
 *  Created on: 31/ott/2019
 *      Author: Antonio&Michela
 */

#include "imagesHandler.h"
#include "../CImg-master/CImg.h"
#include <string>
using namespace cimg_library;

imagesHandler::imagesHandler() {
	// TODO Auto-generated constructor stub
}

imagesHandler::~imagesHandler() {
	// TODO Auto-generated destructor stub
}

std::vector<int> imagesHandler::inputParamAcquisition(char **argi){
	
	//Parameters initialization and acquisition
	std::string fileName = argi[1];
	std::string mode = argi[2];
	std:: string noc = argi[3];
	std:: string iter = argi[4];
	std:: string dispImages =  std::string("not disp");
	if(argi[5]!=NULL)dispImages = argi[5];


	try {
	  std::size_t pos;
	  this->numberOfClusters = std::stoi(noc, &pos);
	  if (pos < noc.size()) {
	    std::cerr << "Trailing characters after number: " << noc << '\n';
	  }
	} catch (std::invalid_argument const &ex) {
	  std::cerr << "Invalid number: " << noc << '\n';
	} catch (std::out_of_range const &ex) {
	  std::cerr << "Number out of range: " << noc << '\n';
	}

	try {
	  std::size_t pos;
	  this->iterations = std::stoi(iter, &pos);
	  if (pos < iter.size()) {
	    std::cerr << "Trailing characters after number: " << iter << '\n';
	  }
	} catch (std::invalid_argument const &ex) {
	  std::cerr << "Invalid number: " << iter << '\n';
	} catch (std::out_of_range const &ex) {
	  std::cerr << "Number out of range: " << iter << '\n';
	}

	try {
	  std::size_t pos;
	  this->dispImages = std::stoi(noc, &pos);
	  if (pos < noc.size()) {
	    std::cerr << "Trailing characters after number: " << noc << '\n';
	  }
	} catch (std::invalid_argument const &ex) {
	  std::cerr << "Invalid number: " << noc << '\n';
	} catch (std::out_of_range const &ex) {
	  std::cerr << "Number out of range: " << noc << '\n';
	}


	std::cout <<"selected file: "<< fileName << std::endl;
	std::cout <<"mode: "<< mode << std::endl;
	std::cout <<"number of clusters: "<< numberOfClusters << " clusters" <<std::endl;
	std::cout <<"iterations: "<< iterations << " iterations"<<std::endl;

	
	this->mode = mode;
	this->fileName = fileName;
	this->dispImages = dispImages;

	int n = fileName.length();
	char file[n + 1];
	std::strcpy(file, fileName.c_str());

	CImg <unsigned char> inputImage = CImg<>(file);

	this-> columns = (int)inputImage.width();
	this-> rows = (int)inputImage.height();


	std::vector<int> params = {this->iterations, this->numberOfClusters, this->columns, this->rows};
	return params;

}



void imagesHandler::dataAcquisition(std::vector<float> &h_x,std::vector<float> &h_y,std::vector<float> &h_z){

	int n = fileName.length();
	char file[n + 1];
	std::strcpy(file, fileName.c_str());
	CImg <unsigned char> inputImage = CImg<>(file);

	if (this->mode.compare("HSV")==0){
		inputImage.RGBtoHSV();
		for(int i=0;i<(this->columns);i++){
				for(int j=0;j<(this->rows);j++){
					h_x[i * (this->rows) + j]=(float)inputImage(i,j,0);
					h_y[i * (this->rows) + j]=(float)(inputImage(i,j,1)*255);
					h_z[i * (this->rows) + j]=(float)(inputImage(i,j,2)*255);
				}
			}
	}

	if (this->mode.compare("sRGB")==0){
		inputImage.RGBtosRGB();
		for(int i=0;i<(this->columns);i++){
				for(int j=0;j<(this->rows);j++){
					h_x[i * (this->rows) + j]=(float)inputImage(i,j,0);
					h_y[i * (this->rows) + j]=(float)inputImage(i,j,1);
					h_z[i * (this->rows) + j]=(float)inputImage(i,j,2);
				}
			}
	}

	if (this->mode.compare("CMY")==0){
		inputImage.RGBtoCMY();
		for(int i=0;i<(this->columns);i++){
				for(int j=0;j<(this->rows);j++){
					h_x[i * (this->rows) + j]=(float)inputImage(i,j,0);
					h_y[i * (this->rows) + j]=(float)inputImage(i,j,1);
					h_z[i * (this->rows) + j]=(float)inputImage(i,j,2);
				}
			}
	}

	if (this->mode.compare("HSL")==0){
		inputImage.RGBtoHSL();
		for(int i=0;i<(this->columns);i++){
				for(int j=0;j<(this->rows);j++){
					h_x[i * (this->rows) + j]=(float)inputImage(i,j,0);
					h_y[i * (this->rows) + j]=(float)(inputImage(i,j,1)*100);
					h_z[i * (this->rows) + j]=(float)(inputImage(i,j,2)*100);
				}
			}
	}

	if (this->mode.compare("HSI")==0){
		inputImage.RGBtoHSI();
		for(int i=0;i<(this->columns);i++){
				for(int j=0;j<(this->rows);j++){
					h_x[i * (this->rows) + j]=(float)inputImage(i,j,0);
					h_y[i * (this->rows) + j]=(float)(inputImage(i,j,1)*100);
					h_z[i * (this->rows) + j]=(float)(inputImage(i,j,2)*100);
				}
			}
	}

	if (this->mode.compare("RGB")==0){
		for(int i=0;i<(this->columns);i++){
			for(int j=0;j<(this->rows);j++){
				h_x[i * (this->rows) + j]=(float)inputImage(i,j,0);
				h_y[i * (this->rows) + j]=(float)inputImage(i,j,1);
				h_z[i * (this->rows) + j]=(float)inputImage(i,j,2);
			}
		}
	}
}

void imagesHandler::disp(int* assignedPixels, std::vector<int> clusterColorR, std::vector<int> clusterColorG, std::vector<int> clusterColorB){

	int n = fileName.length();
	char file[n + 1];
	std::strcpy(file, fileName.c_str());
	CImg <unsigned char> inputImage = CImg<>(file);

	

	CImg<unsigned char> outputImage(this->columns,this->rows,1,3);

	for(int i=0;i<(this->columns);i++){
	  for(int j=0;j<(this->rows);j++){
			outputImage(i,j,0) = clusterColorR[assignedPixels[i*(this->rows)+j]];
			outputImage(i,j,1) = clusterColorG[assignedPixels[i*(this->rows)+j]];
			outputImage(i,j,2) = clusterColorB[assignedPixels[i*(this->rows)+j]];
	  }
	}

	//display loop
	if(this->dispImages.compare("display")==0){

		CImgDisplay draw_disp(this->columns,this->rows,"Clusterized Image");
		CImgDisplay draw_dispO(this->columns,this->rows,"Original Image");

		while (!draw_disp.is_closed() && !draw_disp.is_keyESC() && !draw_disp.is_keyQ()
				&& !draw_dispO.is_closed() && !draw_dispO.is_keyESC() && !draw_dispO.is_keyQ()) {

			inputImage.display(draw_dispO);
			outputImage.display(draw_disp);

			// Temporize event loop
			cimg::wait(20);
		}
	}
	outputImage.normalize(0, 255);

	std::string tmp = this->fileName + "CLUSTERIZED.jpg";
	n = tmp.length();
	char file1[n + 1];
	std::strcpy(file1, tmp.c_str());
	outputImage.save(file1);

}


