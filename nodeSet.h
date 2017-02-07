/*
=== CUDA matrix typedefs/functions v0.2 ===
			By: Austin Bailie

Matrix typedefs with support for higher
dimension matricies.

Adapted from:
	-nVidia's CUDA Programming guide
	-Other credits appear in their respective spots
===========================================
*/
/*
============ Change Log ===================
v0: 1/15/2017		-original

v0.1: 1/21/2017		-added include protection
					-added math.h to support matOps.cu
					-changed Mat2D to support linked lists
					-added laySet to support neural net functionality
					-relocated common functions from matOps.cu to this file

v0.2: 1/29/2017
					-added dTh and dX fields to Mat2D
					-minor additions/changes to common functions

===========================================
*/

//========= Watch your head!! =============
#ifndef __NODESET_H_INCLUDED__
#define __NODESET_H_INCLUDED__

//============== Includes ================
#include "Timer.h"
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <thrust/device_vector.h>

//========== Custom Typedefs =============

typedef struct Mat2D { //row then column
	int rows;
	int rrows;
	int columns;
	int run;
	float* dTh; //new
	float* dX; //new
	float* cells;
	struct Mat2D *next;
	struct Mat2D *end;
} Mat2D;//


/*yeah I made some changes to the higher dimension
  structs... but I haven't needed them yet, so its not worth
  mentioning*/
//typedef struct { //jagged 3d array cells(0) = rows in layer 0, cells(1) = columns  in layer 0, cells(cells(0)*cells(1)+2) = rows in layer 1 and so forth
//	int layers;
//	int count;
//	float* d_nodes;
//	float* cells;
//} jagMat3D;

typedef struct { //row then column then level then time
	int id;
	int rows;
	int columns;
	int levels;
	int time;
	float* cells;
} Mat4D;

typedef struct { //row then column then level then time then fractalPlane
	int id;
	int rows;
	int columns;
	int levels;
	int time;
	int fractalPlane;
	float* cells;
} Mat5D;

typedef struct {
	int* nPl; //an array containing the number of nodes per layer
	int layers; //number of layers
	int bigX;
}laySet;// new

//=========== Classses==============


class Config {
	std::string cfg;
	//bool loaded;
public:
	//Config ();
	Config (std::string);
	~Config();
	std::string in, act, out, timer;
	//void reLoadConfig(std::string);
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
	//std::vector< std::vector<std::string> > lines;
	std::vector< std::vector<std::string> > lines;
	std::vector< std::string > w;
	


	std::ifstream fIn(cfg);

	for (std::string ln; getline(fIn, ln);) {
		std::stringstream s(ln);
		for (std::string wd; getline(s, wd, ',');) {
			w.push_back(wd);
		}
		lines.push_back(w);
		s.clear();//free mem
	}
	fIn.close();
	//setting the different variables from the parsed config file
	in = lines[0][1];
	act = lines[1][1];
	out = lines[2][1];
	batchSize = stoi(lines[3][1]);
	alpha = stof(lines[4][1]);
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


//=========== this code below may be removed/or implemented
//Config::Config (std::string c) {
//	cfg = c;
//	loaded = false;
//}

//void Config::reLoadConfig(std::string c = "") {
//	if (c != "") {
//		cfg = c;
//	}
//	std::vector< std::vector<std::string> > lines;
//	std::vector<std::string> w;
//	std::ifstream fIn(cfg);
//	
//	for (std::string ln; getline(fIn, ln);) {
//		std::stringstream s(ln);
//		for (std::string wd; getline(s, wd, ',');) {
//			w.push_back(wd);
//		}
//		lines.push_back(w);
//		w.clear();
//		s.clear();
//	}
//	fIn.close();
//	in = lines.at(0).at(1);
//	act = lines.at(1).at(1);
//	out = lines.at(2).at(1);
//	batchSize = stoi(lines.at(3).at(1));
//	alpha = stof(lines.at(4).at(1));
//	layers = lines.at(5).size() - 1;
//	nodesPerlayer = (int*)malloc(layers * sizeof(int));
//	for (int i = 0; i < layers; ++i) {
//		nodesPerlayer[i] = stoi(lines.at(5).at(i + 1));
//	};
//	lines.clear;
//}

//======

//using namespace std;


		
//=========== Node Utilities =============

void print2DMat(Mat2D out, const char* prompt = "") { 
	/*simple method to print matrix
	  needs: nodeSet.h, string.h
	  */
	printf("%sMatrix Values:\n{\n", prompt); //just making it pretty
	for (int i = 0; i < out.rows; ++i) { //iterate through each row/col and print
		printf("    "); //again, making pretty
		for (int t = 0; t < out.columns; ++t) {
			printf("%f, ", out.cells[i*out.columns + t]);
		}
		printf("\n");
	}
	printf("}\n");

	printf("%sUpdate Values:\n{\n", prompt); //just making it pretty
	for (int c = 0; c < out.rows; ++c) { //iterate through each row/col and print
		printf("    "); //again, making pretty
		for (int b = 0; b < out.columns; ++b) {
			printf("%f, ", out.dTh[c*out.columns + b]);
		}
		printf("\n");
	}
	printf("}\n");
	//~~ALB
	return;
}

void pprint2DMat(Mat2D* out, const char* prompt = "") { 
	/*simple method to print matrix
	needs: nodeSet.h, string.h
	*/
	printf("%sMatrix Values:\n{\n", prompt); //just making it pretty
	for (int i = 0; i < out->rows; ++i) { //iterate through each row/col and print
		printf("    "); //again, making pretty
		for (int t = 0; t < out->columns; ++t)
			printf("%f, ", out->cells[i*out->columns + t]);
		printf("\n");
	}
	if (out->dTh != NULL){
		printf("}\n");
		printf("%sUpdate Values:\n{\n", prompt); //just making it pretty
		for (int i = 0; i < out->rows; ++i) { //iterate through each row/col and print
			printf("    "); //again, making pretty
			for (int t = 0; t < out->columns; ++t)
				printf("%f, ", out->dTh[i*out->columns + t]);
			printf("\n");
		}
		printf("}\n");
	}
	//~~ALB
}

Mat2D vecToMat2D(float f_vector[], int f_rows, int f_cols) { 
	/*convert vector to a mat2D
	  needs: nodeSet.h, print2DMat(Mat2D out)
	  I don't really need stdlib.h, but malloc shows a pesky error if not... will compile without
	  */

	Mat2D out;//output matrix
	out.rows = f_rows;
	out.columns = f_cols;

	//allocate memory for matrix
	out.cells = (float*)malloc(out.rows*out.columns * sizeof(float));

	//assign values to matrix
	for (int i = 0; i < f_rows; ++i)
		for (int j = 0; j < f_cols; ++j)
			out.cells[i*f_cols + j] = f_vector[i*f_cols + j];
	return out;
	//~~ALB
}

void vecToMat2DP(float f_vector[], Mat2D* inmat) {
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

Mat2D cudaMSend2D(Mat2D iM, bool copy, const char* iD = "matrix") { 
/*Handles GPU memory allocaion/memory transfer to GPU.												
  copy boolean determines if the matrix values should be copied into the allocated memory on GPU
  iD takes a constant char pointer of the matrix name/ID

  Adapted from:
  Robert Hochberg (1/24/16): http://bit.ly/2iA8jDc
*/

//device's copy of the input matrix
	Mat2D d_M;
	//d_M.id = iM.id;
	d_M.rows = iM.rows;
	d_M.columns = iM.columns;

	//allocating memory on GPU for d_M
	cudaError_t errCode = cudaMalloc(&d_M.cells, d_M.rows * d_M.columns * sizeof(float));
	printf("Allocating memory for %s on GPU: %s\n", iD, cudaGetErrorString(errCode));

	//parameter copy decides wheter to copy the iM values to d_M located on GPU
	if (copy) {
		errCode = cudaMemcpy(d_M.cells, iM.cells, d_M.rows * d_M.columns * sizeof(float), cudaMemcpyHostToDevice);
		printf("Copying %s to GPU: %s\n", iD, cudaGetErrorString(errCode));
	}
	return d_M;

	//~~ALB
}

std::istream& csv(std::istream& inp) {
	if ((inp >> std::ws).peek() != std::char_traits<char>::to_int_type(',')) {
		inp.setstate(std::ios_base::failbit);
	}
	return inp.ignore();
}

Mat2D* csvToMat2D(std::string inStr, int cols = 1) {
	std::vector<float> values;
	std::ifstream fin(inStr);
	for (std::string li; std::getline(fin, li);) {
		std::istringstream in;
		in.clear();
		in.str(li);
		for (float val; in >> val; in >> csv) {
			values.push_back(val);
		}
	}

	Mat2D* out = (Mat2D*)malloc(sizeof(Mat2D));
	out->cells = (float*)malloc(values.size() * sizeof(float));
	out->columns = cols;
	out->rows = values.size() / cols;
	for (int i = 0; i < (int)values.size(); ++i) {
		float &t = values.at(i);
		out->cells[i] = t;
	}
	return out;
};
//string csvGetString(string fPath, int lNum, int cNum, int sz = 1) {
//	std::vector<vector<string>> line;
//	std::vector<string> word;
//	string val;
//	Config cfg;
//	bool EX = false;
//	std::ifstream fIn(fPath);
//	int l = 0;
//	for (std::string line; std::getline(fIn, line) && !EX;) {
//		if (l == lNum) {
//			int c = 0;
//			for (std::string word; std::getline(fIn, line, ',');) {
//				if (c == cNum) {
//
//				}
//			}
//		}
//	}
//}

void outToCsv(float* out, Mat2D* act, std::ofstream* outFile, int rn) {
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