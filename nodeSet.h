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
#include <stdio.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
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




#endif