/*
====== CUDA matrix typedefs/functions v1.1 ======
	         		By: Austin Bailie

Matrix typedefs with support for higher
dimension matricies.

Adapted from:
	-nVidia's CUDA Programming guide
	-Other credits appear in their respective spots
=================================================
*/
/*
================= Change Log ====================
v0: 1/15/2017
          -original

v0.1: 1/21/2017		
  -added include protection
	-added math.h to support matOps.cu
	-changed Mat2D to support linked lists
	-added laySet to support neural net functionality
	-relocated common functions from matOps.cu to this file

v0.2: 1/29/2017
	-added dTh and dX fields to Mat2D
	-minor additions/changes to common functions

v1.0: 4/1/2017
  -Changed Mat2D structure to d_Mat2D
  -Implemented Mat2D class
  -Changed malloc to new where appropriate
  -Removed higher dimension matrix structure

v1.1: 4/X/2017: Developing python interface
  -Some re-structuring to address bugs.
=================================================
*/

//========= Watch your head!! =============
#ifndef __NODESET_H_INCLUDED__
#define __NODESET_H_INCLUDED__
//============== Includes ================
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <time.h>
#include "Timer.h"
#include <math.h>
#include <vector>
//========== Custom Typedefs =============
/*
  I can't find a way to allocate the matrices to the gpu without using a
  struct. Passing it as a class makes it fail.
*/
typedef struct d_Mat2D { //row then column
	int rows;
	int columns;
	int run;
	float* dTh = nullptr; //new
	float* dX = nullptr; //new
	float* cells = nullptr;
} d_Mat2D;

typedef struct {
	int* nPl; //an array containing the number of nodes per layer
	int layers; //number of layers
	int bigX;
}LaySet;// new

//=========== Classses==============
class Mat2D {
  std::string label;
public:
  Mat2D();
  Mat2D(int in_rows, int in_cols);
  ~Mat2D();
  enum sub_matrix { Cells = 0, Theta = 1, X = 2 };
  void addHostArrays();
  void resize(int new_rows, int new_cols, sub_matrix matrix, bool hard);
  void gpuDimSet();
  float* gpuMalloc(sub_matrix matrix);
  void gpuSetup();
  void gpuSetup(sub_matrix matrix);
  void gpuSetup(float* send, sub_matrix matrix);
  void gpuSetup(float* to_cells, float* to_dTh, float* to_dX);
  void gpuSend();
  void gpuSend(float* send, float* &mat);
  void gpuRetrieve(sub_matrix matrix);
  void gpuRetrieve(bool free);
  void gpuRetrieve();
  void gpuRetrieve(Mat2D* from, sub_matrix matrix);
  void gpuRetrieve(Mat2D* from);
  void gpuFree();
  int rows = 0;
  int columns = 0;
  int run = 0;
  dim3 tPb;
  dim3 nb;
  float* dTh = nullptr;
  float* dX = nullptr; 
  float* cells = nullptr;
  Mat2D* next = nullptr;
  Mat2D* end =  nullptr; 
  d_Mat2D* dev = nullptr; //Device structure
  unsigned int matSize; //size of matricies of float values
private:
  void setSize(int in_rows, int in_cols);
  float* returnArray(sub_matrix matrix);
  void assignDeviceArray(float* &mat, sub_matrix matrix);
  void initArray(float* &mat);
  void initDeviceStruct();
  void hardResize(int new_rows, int new_cols, float* &change);
  void softResize(int new_rows, int new_cols, float* &change);
};
Mat2D::Mat2D() {

}
Mat2D::Mat2D(int in_rows, int in_cols) {
  setSize(in_rows, in_cols);
  addHostArrays();
  gpuSetup();
}
Mat2D::~Mat2D() {
  //Delete all array members
  if (dTh != nullptr) {
    delete[] dTh;
  }
  if (dX != nullptr) {
    delete[] dX;
  }
  if (cells != nullptr) {
    delete[] cells;
  }
  //Free device struct
  if (dev != nullptr) {
    free(dev);
  }
}
void Mat2D::resize(int new_rows, int new_cols, sub_matrix matrix, 
                                               bool hard = false) {
  float* change = returnArray(matrix);
  if (*change == NULL) hard = true;
  
  if (hard) {
    hardResize(new_rows, new_cols, change);
    return;
  }
  softResize(new_rows, new_cols, change);
  return; 
}
float* Mat2D::returnArray(sub_matrix matrix) {
  if (matrix == Cells) {
    initArray(cells);
    return cells;
  } else if (matrix == Theta) {
    initArray(dTh);
    return dTh;
  } else {
    initArray(dX);
    return dX;
  }
}
void Mat2D::assignDeviceArray(float* &mat, sub_matrix matrix) {
  if (dev == nullptr) {
    //printf("return dev array in first if\n");
    initDeviceStruct();
  }
  if (matrix == Cells) {
    initArray(dev->cells);
    mat = dev->cells;
  } else if (matrix == Theta) {
    initArray(dev->dTh);
    mat = dev->dTh;
  } else {
    initArray(dev->dX);
    mat = dev->dX;
  }
}
void Mat2D::initDeviceStruct() {
  dev = (d_Mat2D*)malloc(sizeof(d_Mat2D));
  dev->rows = rows;
  dev->columns = columns;
}
void Mat2D::initArray(float* &mat) {
  if (mat == NULL) {
    //printf("inreturnInit");
    mat = new float[matSize];
  }
}
void Mat2D::gpuDimSet() {
  dim3 tpb_n(16, 16);
  tPb = tpb_n;
  dim3 nb_n((unsigned int)ceil((double)columns / tPb.x), (unsigned int)ceil((double)rows / tPb.y));
  nb = nb_n;
}
float* Mat2D::gpuMalloc(sub_matrix matrix) {
  float* mat;
  cudaError_t errCode;
  if (dev == nullptr) {
    //printf("return dev array in first if\n");
    initDeviceStruct();
  }
  if (matrix == Cells) {
    errCode = cudaMalloc(&dev->cells, matSize);
    mat = dev->cells;
  } else if (matrix == Theta) {
    errCode = cudaMalloc(&dev->dTh, matSize);
    mat = dev->dTh;
  } else {
    errCode = cudaMalloc(&dev->dX, matSize);
    mat = dev->dX;
  }
  printf("GPU cudaMalloc: %s\n", cudaGetErrorString(errCode));
  return mat;
}
void Mat2D::gpuSetup() {
  gpuSetup(Cells);
  gpuSetup(Theta);
  gpuSetup(X);
}
void Mat2D::gpuSetup(sub_matrix matrix) { 
  float* send = returnArray(matrix);
  gpuSetup(send, matrix);
}
void Mat2D::gpuSetup(float* send, sub_matrix matrix) {
  float* d_mat = gpuMalloc(matrix);
  gpuDimSet();
  gpuSend(send, d_mat);
}
void Mat2D::gpuSetup(float* to_cells, float* to_dTh, float* to_dX) {
  gpuSetup(to_cells, Cells);
  gpuSetup(to_dTh, Theta);
  gpuSetup(to_dX, X);
}
void Mat2D::gpuSend() {
  gpuSend(cells, dev->cells);
  gpuSend(dTh, dev->dTh);
  gpuSend(dX, dev->dX);
}
void Mat2D::gpuSend(float* send, float* &mat) {
  cudaError_t errCode = cudaMemcpy(mat, send, matSize, cudaMemcpyHostToDevice);
  printf("Memcpy: %s\n", cudaGetErrorString(errCode));
}
void Mat2D::gpuRetrieve(sub_matrix matrix) {
  float* d_mat;
  assignDeviceArray(d_mat, matrix);
  float* mat = returnArray(matrix);
  //cudaError_t errCode = cudaMemcpy(mat, d_mat, matSize, cudaMemcpyDeviceToHost);
  //printf("Retrieving nodes from GPU: %s\n", cudaGetErrorString(errCode));
}
void Mat2D::gpuRetrieve(bool free) {
  gpuRetrieve();
  if (free) gpuFree();
}
void Mat2D::gpuRetrieve() {
  gpuRetrieve(Cells);
  gpuRetrieve(Theta);
  gpuRetrieve(X);
}
void Mat2D::gpuRetrieve(Mat2D* from, sub_matrix matrix) {
  from->gpuRetrieve(matrix);
  float* mat = returnArray(matrix);
  *mat = *from->returnArray(matrix);
}
void Mat2D::gpuRetrieve(Mat2D* from) {
  gpuRetrieve(from, Cells);
  gpuRetrieve(from, Theta);
  gpuRetrieve(from, X);
}
void Mat2D::gpuFree() {
  cudaFree(dev->cells);
  cudaFree(dev->dTh);
  cudaFree(dev->dX);
}
void Mat2D::setSize(int in_rows, int in_cols) {
  rows = in_rows;
  columns = in_cols;
  matSize = rows * columns * sizeof(float);
}
void Mat2D::addHostArrays() {
  cells = new float[matSize];
  dTh = new float[matSize];
  dX = new float[matSize];
}
void Mat2D::hardResize(int new_rows, int new_cols, float* &change) {
  delete change;
  setSize(new_rows, new_cols);
  change = new float[matSize];
}
void Mat2D::softResize(int new_rows, int new_cols, float* &change) {
  setSize(new_rows, new_cols);
  float* temp;
  temp = new float[matSize];
  for (int i = 0; i < sizeof(*change) / sizeof(float) && i < rows * columns;
                                                                         ++i) {
    temp[i] = change[i];
  }
  delete change;
  change = temp;
}
Mat2D* newMat2D(int in_rows, int in_cols) {
  Mat2D *mat_ptr = new Mat2D(in_rows, in_cols);
  return mat_ptr;
}
// Configuration class that holds the primary settings for the neural network.
class Config {
	std::string cfg;
	//bool loaded;
public:
	// Config ();
	Config (std::string);
	~Config();
	std::string in, act, out, timer;
	// void reLoadConfig(std::string);
	int* nodesPerlayer;
	int layers, batchSize;
	float alpha;

};
//define constructor
Config::Config (std::string c = "") {
	if (c != "") {
		cfg = c;
	}
	else {
		cfg = "setup.csv";
	}
	//Creating 2d vector of the setup file for easy access
	std::vector< std::vector<std::string> > lines;
	std::vector< std::string > w;
	std::ifstream fIn(cfg);
	for (std::string ln; getline(fIn, ln);) {
		std::stringstream s(ln);
		for (std::string wd; getline(s, wd, ',');) {
			w.push_back(wd);
		}
		lines.push_back(w);
		s.clear();
		w.clear();
	}
	fIn.close();
	//setting the different variables from the parsed config file
	in = lines[0][1];
	act = lines[1][1];
	out = lines[2][1];
	batchSize = std::stoi(lines[3][1]);
	alpha = std::stof(lines[4][1]);
	layers = lines[5].size() - 1;
	nodesPerlayer = (int*)malloc(layers * sizeof(int));
	for (int i = 0; i < layers; ++i) {
		nodesPerlayer[i] = stoi(lines[5][i + 1]);
	};

	if (lines[6].size() > 1){
		timer = lines[6][1];
	}
	else {
		timer = "";
	}
	//free some memory
	lines.clear();
}
//define destructor
Config::~Config() {
	//delete the allocated memory for nodesPerLayer
	free(nodesPerlayer);
}



		
//=========== Node Utilities =============

void Print2DMatrix(Mat2D* matrix, const char* label = "") { 
	/*simple method to print matrix
	needs: nodeSet.h, string.h
	*/
	printf("%sMatrix Values:\n{\n", label); //just making it pretty
	for (int i = 0; i < matrix->rows; ++i) { //iterate through each row/col and print
		printf("    "); //again, making pretty
		for (int t = 0; t < matrix->columns; ++t)
			printf("%f, ", matrix->cells[i*matrix->columns + t]);
		printf("\n");
	}
	if (matrix->dTh != NULL){
		printf("}\n");
		printf("%sUpdate Values:\n{\n", label); //just making it pretty
		for (int i = 0; i < matrix->rows; ++i) { //iterate through each row/col and print
			printf("    "); //again, making pretty
			for (int t = 0; t < matrix->columns; ++t)
				printf("%f, ", matrix->dTh[i*matrix->columns + t]);
			printf("\n");
		}
		printf("}\n");
	}
	fflush(stdout);
	//~~ALB
}

Mat2D ArrayToMat2D(float in_array[], int in_rows, int in_columns) { 
	/*convert vector to a mat2D
	  needs: nodeSet.h, print2DMat(Mat2D out)
	  I don't really need stdlib.h, but malloc shows a pesky error if not... will compile without
	  */

	Mat2D out;//output matrix
	out.rows = in_rows;
	out.columns = in_columns;

	//allocate memory for matrix
	out.cells = (float*)malloc(out.rows*out.columns * sizeof(float));

	//assign values to matrix
	for (int i = 0; i < in_rows; ++i)
		for (int j = 0; j < in_columns; ++j)
			out.cells[i*in_columns + j] = in_array[i*in_columns + j];
	return out;
	//~~ALB
}

void ArrayPointerToMat2D(float f_vector[], Mat2D* inmat) {
	/*convert vector to a mat2D... this is a function for use on the pointers similar to the one above
	needs: nodeSet.h, print2DMat(Mat2D out)
	I don't really need stdlib.h, but malloc shows a pesky error if not... will compile without
	*/


	//allocate memory for matrix
	inmat->cells = (float*)malloc(inmat->rows * inmat->columns * sizeof(float));

	//assign values to matrix
	for (int i = 0; i < inmat->rows; ++i)
		for (int j = 0; j <  inmat->columns; ++j)
			inmat->cells[i*inmat->columns + j] = f_vector[i*inmat->columns + j];
	return;
	//~~ALB
}

Mat2D CudaSendMat2D(Mat2D input_matrix, bool copy, const char* label = "matrix") { 
/*Handles GPU memory allocaion/memory transfer to GPU.												
  copy boolean determines if the matrix values should be copied into the allocated memory on GPU
  iD takes a constant char pointer of the matrix name/ID

  Adapted from:
  Robert Hochberg (1/24/16): http://bit.ly/2iA8jDc
*/

//device's copy of the input matrix
	Mat2D d_matrix;
	//d_M.id = iM.id;
	d_matrix.rows = input_matrix.rows;
	d_matrix.columns = input_matrix.columns;

	//allocating memory on GPU for d_M
	cudaError_t errCode = cudaMalloc(&d_matrix.cells, d_matrix.rows * d_matrix.columns * sizeof(float));
	printf("Allocating memory for %s on GPU: %s\n", label, cudaGetErrorString(errCode));

	//parameter copy decides wheter to copy the iM values to d_M located on GPU
	if (copy) {
		errCode = cudaMemcpy(d_matrix.cells, input_matrix.cells, d_matrix.rows * d_matrix.columns * sizeof(float), cudaMemcpyHostToDevice);
		printf("Copying %s to GPU: %s\n", label, cudaGetErrorString(errCode));
	}
	return d_matrix;

	//~~ALB
}

std::istream& Csv(std::istream& inp) {
	if ((inp >> std::ws).peek() != std::char_traits<char>::to_int_type(',')) {
		inp.setstate(std::ios_base::failbit);
	}
	return inp.ignore();
}

Mat2D* CsvToMat2D(std::string input_string, int columns = 1) {
	std::vector<float> values;
	std::ifstream fin(input_string);
	for (std::string li; std::getline(fin, li);) {
		std::istringstream in;
		in.clear();
		in.str(li);
		for (float val; in >> val; in >> Csv) {
			values.push_back(val);
		}
	}

  Mat2D* out = newMat2D(values.size() / columns, columns);//(Mat2D*)malloc(sizeof(Mat2D));
	//out->cells = (float*)malloc(values.size() * sizeof(float));
	//out->columns = columns;
	//out->rows = values.size() / columns;
	for (int i = 0; i < (int)values.size(); ++i) {
		float &t = values.at(i);
		out->cells[i] = t;
	}
	return out;
};


void ArrToCsv(float* out, Mat2D* act, std::ofstream* outFile, int rn) {
	float errSq;
	int c = act->columns;
	*outFile << std::endl;
	for (int i = 0; i < c; ++i) {
		errSq = (out[i] - act->cells[i]);
		errSq = errSq * errSq;
		*outFile << out[i] << ", " << act->cells[rn * c + i] << ", " << errSq;
	}
	return;
}

#endif