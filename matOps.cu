/*
======== CUDA matrix operations v0.2 ========
			By: Austin Bailie

Various matrix operations computed on the GPU.
Over-commented for educational purposes

Adapted from:
	-nVidia's CUDA Programming guide
	-Other credits appear in their respective spots
===========================================
*/
/*
============ Change Log ===================
v0: 1/15/2017		
                  - original

v0.01: 1/15/2017	
                  - added transpose

v0.1: 1/21/2016		
                  - added various matrix math functions
					        - added neural network architecture:
						      - added logistic2D kernel
						      - added lmatSend2D
						      - added nodeRetrieve
						      - added processNodes
						      - added layersetup
						      - added hiddenSetup
				        	- current neural network support runs with 0 errors on Cuda-memcheck

v0.2: 1/29/2017		- implemented working neural network functionality:
						      - added nodeBackwardLog kernel
						      - added outPivotLog kernel
						      - added updateNodes kernel
						      - added sendActual support function
						      - addded uNodes support function
						      - added changeIn support function
						      - added pNodes process function
						      - tweaked most of the previously added neural network architecture
						      - changed main function to run the neural network
					        - current technology will successfuly perform batch gradient descent
					          with 0 errors on Cuda-memcheck

v1.0: 4/1/2017
                  - Changed the code in this module to support the new Mat2D class

===========================================
*/
//required headers:
#include "nodeSet.h"  
#include <random>
#include <thread>
using namespace std;

#define BLKSZ 16
//int globalSeed;
/////////////////////////////////////////////////////////////////////////////80
/*CUDA kernel for 2d matrix addition
*/
__global__ void MatrixAddKernel(d_Mat2D d_matrix_a, d_Mat2D d_matrix_b, 
                                d_Mat2D d_output) {
	int r = blockDim.y * blockIdx.y + threadIdx.y; //getting row based on CUDA thread/block index
	int c = blockDim.x * blockIdx.x + threadIdx.x; //getting column based on CUDA thread/block index
	
	//if out-of-bounds stop
	if (r > d_output.rows || c > d_output.columns) return;

	//add
	d_output.cells[r * d_output.columns + c] = d_matrix_a.cells[r * d_output.columns + c] + d_matrix_b.cells[r * d_output.columns + c];
}

/*CUDA kernel for 2d matrix subtraction
*/
__global__ void MatrixSubtractKernel(d_Mat2D d_matrix_a, d_Mat2D d_matrix_b, 
                                     d_Mat2D d_output) {
	int r = blockDim.y * blockIdx.y + threadIdx.y; //getting row based on CUDA thread/block index
	int c = blockDim.x * blockIdx.x + threadIdx.x; //getting column based on CUDA thread/block index

	//if out-of-bounds stop
	if (r > d_output.rows || c > d_output.columns) return;

	//subtract
	d_output.cells[r * d_output.columns + c] = d_matrix_a.cells[r * d_output.columns + c] - d_matrix_b.cells[r * d_output.columns + c];
}

/*CUDA kernel for 2d matrix subtraction
*/
__global__ void MatrixTransposeKernel(d_Mat2D d_input, d_Mat2D d_output) {
	int r = blockIdx.y * blockDim.y + threadIdx.y; //getting row based on CUDA thread/block index
	int c = blockIdx.x * blockDim.x + threadIdx.x; //getting column based on CUDA thread/block index

	//if out-of-bounds stop
	if (r > d_output.rows || c > d_output.columns) return;
	
	//switch rows and columns
	d_output.cells[r * d_output.columns + c] = d_input.cells[c * d_input.columns + r];
}

/*CUDA kernel for 2D matrix multiplication
Adapted from:
-Robert Hochberg (1/24/16): http://bit.ly/2iA8jDc
*/
__global__ void MatrixMultiplyKernel(d_Mat2D d_matrix_a, d_Mat2D d_matrix_b, 
                                     d_Mat2D d_c) { 
	float oVal = 0;//output value

	int r = blockIdx.y * blockDim.y + threadIdx.y; //getting row based on CUDA thread/block index
	int c = blockIdx.x * blockDim.x + threadIdx.x; //getting column based on CUDA thread/block index

	//add up each A(r,i)*B(i,c) into oVal
	if (r < d_c.rows && c < d_c.columns) {
		for (int i = 0; i < d_matrix_a.columns; ++i)
			oVal += d_matrix_a.cells[r * d_matrix_a.columns + i] * d_matrix_b.cells[i*d_matrix_b.columns + c];
	}
	else {
		return;
	}
	//assign the oVal to the output matrix
	d_c.cells[r * d_c.columns + c] = oVal;
}

/*CUDA kernel/device function for logistic nodes:
This is the main forward (input to output) function.	This function is going
forward on the recursion path of the host function
*/
__device__ void LogNodeFunctionDevice(d_Mat2D d_layer, d_Mat2D d_prev, int row, 
                                          int col, int max_row, int max_col) {
	float l = d_layer.cells[max_col + col];//add in the bias
	/* We need to add up all of x*w of the previous layer, where:
	   x = the output of a node in the previous layer, stored in row 1 of the 
         previous layer.
	   w = the weight of the corresponding output, stored in column c of the 
         current layer */
	for (int i = 2; i < max_row;) { 
    // Start at 2 since 0 has the outputs and 1 has the bias
		l += d_layer.cells[i * max_col + col] * d_prev.cells[i - 2];
		++i;
	}

	d_layer.cells[col] = 1 / (1 + exp(-l)); //now we calculate the output of this node
	printf("Row: %i, Col: %i, Input: %f, Value: %f \n", row, col, l, d_layer.cells[col]);
  }

__global__ void LogNodeFunctionKernel(d_Mat2D d_layer, d_Mat2D d_prev) {
	int row = blockDim.y * blockIdx.y + threadIdx.y; 
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	//printf("blockIdx.x %i, blockIdx.y %i\n", blockIdx.x, blockIdx.y);
	int max_row = d_layer.rows;
	int max_col = d_layer.columns;
	//printf("r %i, lr %i, c %i, lc %i, bIx %i, bIy %i, tIx %i, tIy %i\n", r, 
  //        lr, c, lc, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
	if (row < 1 && col < max_col) {
		LogNodeFunctionDevice(d_layer, d_prev, row, col, max_row, max_col);
	}
	else {
		return;
	}
}


__device__ float RowSumDxDevice(d_Mat2D d_matrix_a, int r) {
	float out = 0;
	int cc = d_matrix_a.columns;
	for (int i = 0; i < cc; ++i) {
		out += d_matrix_a.dX[r * cc + i];
	}
	return out;
}

__device__ void LogisticBackpropagationDevice(d_Mat2D d_previous, d_Mat2D d_current, 
                                              d_Mat2D d_next, int row, int col, 
                                              int max_row, int max_col) {
	/*CUDA kernel for back propagation on logistic nodes
	the .dTh matrix is a matrix of the change in weights for the corresponding node-input pair
	the .dX matrix is a matrix of the change in the cascaded error term for the corresponding node-input pair
	the first row of .dX holds the sum  of the .dX terms in the next layer of the corresponding input

	This code is going backward (along the return path of the host recursive function)
	needs: nodeset.h, math.h
	*/

	if (row > 0 && row < max_row && col < max_col) { //only operating on the weights and bias
		//printf("%f", d_current.dX[c]); //debug code
		
		float tX = d_current.cells[col]; //initialize variable for the dX/dL term
		float bX;
		bX = tX - (tX * tX); //the (x- x^2) term of dX/dL
		
		//printf("If 1 True\nR: %i, C: %i\n", r, c); //debug code
		//printf("d_next.columns %i\n", d_next.columns); //debug code

		float DX = d_current.dX[col]; //the total dX/dx of the next layer is sumed into this layer in the first row of the .dX matrix
		//printf(" dbX_n %f ", dbX_n); //debug code

		if (row == 1) { //change to bias
			float dB = d_current.dTh[row * max_col + col];
			dB = dB + bX * DX;
			d_current.dTh[row * max_col + col] = dB;
			//d_current.dX[d_current.columns + c] = dbX_n;
		}
		else { //change to the weights (theta)
			float dTh = d_current.dTh[row * max_col + col];
			float pX = d_previous.cells[row-2];
			dTh += DX * bX * pX;
			d_current.dTh[row * max_col + col] = dTh;
			d_current.dX[row * max_col + col] = d_current.dX[col] * bX * d_current.cells[row * max_col + col];

			//d_prev.dX[r - 2] = d_prev.dX[r - 2] + DX*bX*d_current.cells[r * d_current.columns + c]; //sum the dX/dx layer for the previous layer... this needs to go after syncthreads
		}
	}
	else {
		return;
	}
}

__global__ void LogisticBackpropagationKernel(d_Mat2D d_prev, d_Mat2D d_cur, 
                                              d_Mat2D d_next) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int cr = d_cur.rows;
	int cc = d_cur.columns;
	LogisticBackpropagationDevice(d_prev, d_cur, d_next, r, c, cr, cc);
	__syncthreads();

	if (r > 1 && r - 2 < d_prev.columns && c == 0) {
		float p = RowSumDxDevice(d_cur, r);
		d_prev.dX[r - 2] = d_prev.dX[r - 2] + p;
	}
	else {
		return;
	}

}

/*CUDA kernel/device function for error comparison on logistic nodes:

The .dTh matrix is a matrix of the change in weights for the corresponding 
node-input pair. The .dX matrix is a matrix of the change in the cascaded error
term for the corresponding node-input pair. The first row of .dX holds the sum
of the .dX terms in the next layer of the corresponding input

This code is the pivot of the recursive function (aka from going forward to going backward)
*/
__device__ void LogisticOutputLayerPivotDevice(d_Mat2D d_previous, d_Mat2D d_current, 
                                               d_Mat2D d_actual, int run, int row, 
                                               int col, int max_row, int max_col) {

	if (row > 0 && row < max_row && col < max_col) {
		//printf("If 1 True\nR: %i, C: %i\n", r, c); //debug code
		//printf("d_cur.columns %i\n", d_cur.columns); //debug code
		float bX = d_current.cells[col];
		bX = bX - (bX * bX);
		//printf("piv c: %i, run: %i, actual: %f\n", c, rn, actual.cells[rn * actual.columns + c]); //debug code
		float act = d_actual.cells[run * d_actual.columns + col];
		float err = d_current.cells[col] - act;
		printf("Error Squared: %f\n, actual: %f\n", err * err, act);
		//printf("pre if 2\n");
		
		if (row == 1) { //change to bias
			//printf("True-2\n"); //debug code
			//printf("d_cur.columns %i", d_cur.columns); //debug code
			float dB = d_current.dTh[row * max_col + col];
			dB = dB + bX * err;
			if (col == 0) {
				d_current.dX[max_col] += err*err;
			}
			//printf("dTh(%i, %i): %f, dX: %f\n", r, c, d_cur.dTh[r * d_cur.columns + c], d_cur.dX[d_cur.columns + c]); //debug code
		}
		else { //change to the weights (theta)
			//printf("Else-2\n"); //debug code
			float dTh = d_current.dTh[row * max_col + col];
			float pX = d_previous.cells[row-2];
			dTh += err * bX * pX;
			d_current.dTh[row * max_col + col] = dTh;
			d_current.dX[row *max_col + col] = err * bX * d_current.cells[row * max_col + col];
		}
	}
	else
	{
		return;
	}
}

__global__ void LogisticOutputLayerPivotKernel(d_Mat2D d_previous, d_Mat2D d_current, 
                                               d_Mat2D actual, int run) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int cr = d_current.rows;
	int cc = d_current.columns;
	LogisticOutputLayerPivotDevice(d_previous, d_current, actual, run, r, c, cr, cc);
	__syncthreads();

	if (r > 1 && r - 2 < d_previous.columns && c == 0) {
		float p = RowSumDxDevice(d_current, r);
		d_previous.dX[r - 2] = d_previous.dX[r - 2] + p;
	}
	else {
		return;
	}

}

/*CUDA kernel for weight/bias updates
This node is the update step of the learning process
*/
__global__ void UpdateNodesKernel(d_Mat2D d_nodes, float alpha) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int cr = d_nodes.rows;
	int cc = d_nodes.columns;
	//printf("Node Update: R- %i, C- %i, dnodes r- %i, dnodes c- %i \n", r, c, d_nodes.rows, d_nodes.columns);

	if (r > 0 && r < cr && c < cc) { //only update weights/biases
		float cur = d_nodes.cells[r * cc + c];
		float del = d_nodes.dTh[r * cc + c];
		d_nodes.cells[r * cc + c] = cur - (alpha * del);
		//printf(" %f \n", del); //debug code
		d_nodes.dTh[r * cc + c] = 0;
		d_nodes.dX[r * cc + c] = 0;
	}
	else if (r==0 & r < cr && c < cc) {
		d_nodes.dTh[r * cc + c] = 0;
		d_nodes.dX[r * cc + c] = 0;
		/*if (r == 0 & c == 0) {
			d_nodes.dX[cc] = 0;
		}*/
	}
	else {
		return;
	}
}


__global__ void ChangeInputsKernel(d_Mat2D d_nodes, d_Mat2D d_in, int run) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (r == 0 && c < d_nodes.columns) {
		printf("Layer 0: R: %i, C: %i, in: %f \n", r, c, d_in.cells[run * d_in.columns + c]);
		d_nodes.cells[c] = d_in.cells[run * d_in.columns + c];
	}
	else {
		return;
	}
}

//Mat2D cudaMatrixAdd2D(Mat2D f_mA, Mat2D f_mB) {
//	if (f_mA.columns != f_mB.columns || f_mA.rows != f_mB.rows) {
//		printf("ERROR: Incorrect array dimensions A + B. Sizes must be equal.\n");
//    printf("A: %i x %i, B: %i x %i\n", f_mA.rows, f_mA.columns, f_mB.rows, f_mB.columns);
//		Mat2D err;
//		err.rows = -1;
//		return err;
//	}
//	//Send input matricies to GPU and return d_mA so GPU memory can be deallocated later
//	printf("--------- Addition ---------\n");
//	Mat2D d_mA = CudaSendMat2D(f_mA, true, "matrix A");
//	Mat2D d_mB = CudaSendMat2D(f_mB, true, "matrix B");
//
//	//Create output matrix and allocate memory on GPU. Returns d_out to access result/deallocate mem
//	Mat2D out;
//	out.rows = d_mA.rows;
//	out.columns = d_mA.columns;
//	out.cells = (float*)malloc(out.rows * out.columns * sizeof(float));
//	Mat2D d_out = CudaSendMat2D(out, false, "Output matrix");
//
//	//setup CUDA architecture and run kernel
//	dim3 threadsPerBlock(16, 16); //each block will contain 16 by 16 threads
//	dim3 numBlocks((d_out.columns + threadsPerBlock.x - 1) / threadsPerBlock.x, //number of blocks on x dimension of grid
//		(d_out.rows + threadsPerBlock.y - 1) / threadsPerBlock.y); //number of blocks on y dimension of grid
//	MatrixAddKernel <<< numBlocks, threadsPerBlock >>> (d_mA, d_mB, d_out); //run's kernal
//	cudaError_t errCode = cudaThreadSynchronize(); //synchronize cores to ensure everthing has been run
//	printf("GPU Thread Synchronization: %s\n", cudaGetErrorString(errCode));
//
//	//debug code to find errors in execution of kernel
//	errCode = cudaGetLastError();
//	if (errCode != cudaSuccess)
//	{
//		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(errCode));
//		exit(-1);
//	}
//
//	//retrieve output matrix from GPU memory
//	errCode = cudaMemcpy(out.cells, d_out.cells, d_out.rows * d_out.columns * sizeof(float), cudaMemcpyDeviceToHost);
//	printf("Pulling Output matrix from GPU: %s\n", cudaGetErrorString(errCode));
//
//	//print result
//	Print2DMatrix(out, "\n\n--Addition Results--\nOutput ");
//	printf("--------------------------------\n");
//
//	//deallocate GPU memory
//	cudaFree(d_mA.cells);
//	cudaFree(d_mB.cells);
//	cudaFree(d_out.cells);
//
//	return out;
//}

//Mat2D cudaMatrixSubtract2D(Mat2D f_mA, Mat2D f_mB) {
//	if (f_mA.columns != f_mB.columns || f_mA.rows != f_mB.rows) {
//		printf("ERROR: Incorrect array dimensions A + B. Sizes must be equal.\n");
//		printf("A: %i x %i, B: %i x %i\n", f_mA.rows, f_mA.columns, f_mB.rows, f_mB.columns);
//		Mat2D err;
//		err.rows = -1;
//		return err;
//	}
//	//Send input matricies to GPU and return d_mA so GPU memory can be deallocated later
//	printf("--------- Subtraction ---------\n");
//	Mat2D d_mA = CudaSendMat2D(f_mA, true, "matrix A");
//	Mat2D d_mB = CudaSendMat2D(f_mB, true, "matrix B");
//
//	//Create output matrix and allocate memory on GPU. Returns d_out to access result/deallocate mem
//	Mat2D out;
//	out.rows = d_mA.rows;
//	out.columns = d_mA.columns;
//	out.cells = (float*)malloc(out.rows * out.columns * sizeof(float));
//	Mat2D d_out = CudaSendMat2D(out, false, "Output matrix");
//
//	//setup CUDA architecture and run kernel
//	dim3 threadsPerBlock(16, 16); //each block will contain 16 by 16 threads
//	dim3 numBlocks((d_out.columns + threadsPerBlock.x - 1) / threadsPerBlock.x, //number of blocks on x dimension of grid
//		(d_out.rows + threadsPerBlock.y - 1) / threadsPerBlock.y); //number of blocks on y dimension of grid
//	MatrixSubtractKernel <<<numBlocks, threadsPerBlock >>> (d_mA, d_mB, d_out); //run's kernal
//	cudaError_t errCode = cudaThreadSynchronize(); //synchronize cores to ensure everthing has been run
//	printf("GPU Thread Synchronization: %s\n", cudaGetErrorString(errCode));
//
//	//debug code to find errors in execution of kernel
//	errCode = cudaGetLastError();
//	if (errCode != cudaSuccess)
//	{
//		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(errCode));
//		exit(-1);
//	}
//
//	//retrieve output matrix from GPU memory
//	errCode = cudaMemcpy(out.cells, d_out.cells, d_out.rows * d_out.columns * sizeof(float), cudaMemcpyDeviceToHost);
//	printf("Pulling Output matrix from GPU: %s\n", cudaGetErrorString(errCode));
//
//	//print result
//	Print2DMatrix(out, "\n\n--Subtraction Results--\nOutput ");
//	printf("--------------------------------\n");
//
//	//deallocate GPU memory
//	cudaFree(d_mA.cells);
//	cudaFree(d_mB.cells);
//	cudaFree(d_out.cells);
//
//	return out;
//}

//Mat2D cudaMatrixTranspose2D(Mat2D f_m) {
//	printf("--------- Transposition ---------\n");
//	Mat2D d_matrix = CudaSendMat2D(f_m, true, "Original");
//	Mat2D out = Mat2D(f_m.columns, f_m.rows);
//	Mat2D d_out = CudaSendMat2D(out, true, "output");
//
//	dim3 tPBlock(16, 16);
//	dim3 nBlocks((out.columns + tPBlock.x - 1) / tPBlock.x, (out.rows + tPBlock.y - 1) / tPBlock.y);
//	MatrixTransposeKernel << <nBlocks, tPBlock >> > (d_matrix, d_out);
//	cudaError_t errCode = cudaThreadSynchronize(); //synchronize cores to ensure everthing has been run
//	printf("GPU Thread Synchronization: %s\n", cudaGetErrorString(errCode));
//
//	//debug code to find errors in execution of kernel
//	errCode = cudaGetLastError();
//	if (errCode != cudaSuccess)
//	{
//		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(errCode));
//		exit(-1);
//	}
//
//	//retrieve output matrix from GPU memory
//	errCode = cudaMemcpy(out.cells, d_out.cells, 
//                       d_out.rows * d_out.columns * sizeof(float), 
//                                                      cudaMemcpyDeviceToHost);
//	printf("Pulling Output matrix from GPU: %s\n", cudaGetErrorString(errCode));
//
//	//print result
//	Print2DMatrix(out, "\n\n--Transposition Results--\nOutput ");
//	printf("--------------------------------\n");
//
//	cudaFree(d_matrix.cells);
//	cudaFree(d_out.cells);
//
//	return out;
//}

//Mat2D cudaMatrixMult2D(Mat2D f_mA, Mat2D f_mB) { 
//	/*2D Matrix multiplication algorithm
//	  needs: mMultKernel2D<<<nBlocks, tPb>>>(Mat2D d_mA, Mat2D d_mB, Mat2D d_out),
//	  cudaMSend2D(Mat2D f_mX, Bool TF, const char* ID), nodeSet.h
//
//	  Adapted from:
//	  -Robert Hochberg (1/24/16): http://bit.ly/2iA8jDc
//	  */
//
//	//check dimensions, err if incorrect
//	if (f_mA.columns != f_mB.rows) {
//		printf("ERROR: Incorrect array dimensions A*B. Number of columns in A must equal number of rows in B.\n");
//		printf("A: %i x %i, B: %i x %i\n", f_mA.rows, f_mA.columns, f_mB.rows, f_mB.columns);
//		Mat2D err;
//		err.rows = -1;
//		return err;
//	}
//
//	//Send input matricies to GPU and return d_mA so GPU memory can be deallocated later
//	printf("---------Multiplication---------\n");
//	Mat2D d_mA = CudaSendMat2D(f_mA, true, "matrix A");
//	Mat2D d_mB = CudaSendMat2D(f_mB, true, "matrix B");
//
//	//Create output matrix and allocate memory on GPU. Returns d_out to access result/deallocate mem
//	Mat2D out;
//	//out.id = getID(master, 1);
//	out.rows = d_mA.rows;
//	out.columns = d_mB.columns;
//	out.cells = (float*)malloc(out.rows * out.columns * sizeof(float));
//	Mat2D d_out = CudaSendMat2D(out, false, "Output matrix");
//
//	//setup CUDA architecture and run kernel
//	dim3 threadsPerBlock(16, 16); //each block will contain 16 by 16 threads
//	dim3 numBlocks((d_out.columns + threadsPerBlock.x - 1) / threadsPerBlock.x, //number of blocks on x dimension of grid
//		(d_out.rows + threadsPerBlock.y) / threadsPerBlock.y); //number of blocks on y dimension of grid
//	MatrixMultiplyKernel << <numBlocks, threadsPerBlock >> > (d_mA, d_mB, d_out); //run's kernal
//	cudaError_t errCode = cudaThreadSynchronize(); //synchronize cores to ensure everthing has been run
//	printf("GPU Thread Synchronization: %s\n", cudaGetErrorString(errCode));
//
//	//debug code to find errors in execution of kernel
//	errCode = cudaGetLastError();
//	if (errCode != cudaSuccess)
//	{
//		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(errCode));
//		exit(-1);
//	}
//
//	//retrieve output matrix from GPU memory
//	errCode = cudaMemcpy(out.cells, d_out.cells, d_out.rows * d_out.columns * sizeof(float), cudaMemcpyDeviceToHost);
//	printf("Pulling Output matrix from GPU: %s\n", cudaGetErrorString(errCode));
//
//	//print result
//	Print2DMatrix(out, "\n\n--Multiplication Results--\nOutput ");
//	printf("--------------------------------\n");
//
//	//deallocate GPU memory
//	cudaFree(d_mA.cells);
//	cudaFree(d_mB.cells);
//	cudaFree(d_out.cells);
//
//	return out;
//}

//int intRand(const int & min, const int & max) {
//  static thread_local mt19937 generator;
//  uniform_int_distribution<int> distribution(min, max);
//  return distribution(generator);
//}
float intRand(float min, float max) {
  static thread_local mt19937 generator;
  uniform_int_distribution<int> distribution(0, 100);
  float range = max - min;
  float out = (float)distribution(generator) * (float)range / 100.;
  out += min;
  return out;
}

Mat2D* layerSetup(LaySet setup, int indx, bool onesFirst = true) { 
	/*Setup of the node layers:
	  Each array has the output node in row zero, and the weights for each input
    in the rows below. So, # of rows = # of nodes from previous layer + 2 
    (the output node row and a bias row on row 1). This also means, that the 
    number of rows equals the number of columns of the previous layer + 2
	*/
	Mat2D* out = newMat2D(2, setup.nPl[indx]);
	/* The first layer will have no weight or bias, but has to have 2 rows	*/
	if (indx == setup.layers - 1 || indx != 0) {
    out->resize(setup.nPl[indx - 1] + 2, out->columns, out->Cells);
	}
  out->addWeightArray();

	//initialize array with 1's in row 0 and zeros in rest
	float zer = 0.;
	float bz = 1.;
	for (int r = 0; r < out->rows; ++r) {
		for (int c = 0; c < out->columns; ++c) {
      float val = pow(intRand(-2, 2) / out->rows, zer)*bz;
      if (val > 0.99) {
        out->cells[r * out->columns + c] = 0.99;
      } else {
        out->cells[r * out->columns + c] = val;
      }
			out->dTh[r * out->columns + c] = 0;
			out->dX[r * out->columns + c] = 0;
		}
		bz = 1;
		if (r == 0) bz = 0;
		zer = 1;
	}
	return out;
}

Mat2D* hiddenSetup(LaySet setup) {
	/* Setup of the 'hidden' layers and input layer
	   The goal is to setup a linked list. Its the dream..
	   From here on we're passing pointers rather than the structure
	*/
	printf("\n\n============= Hidden Layers Setup ===============\n");
	
	/* The below code is my way of setting up a linked list. 
	   I think there's a better way to code this without dereferencing so much,
	   but it works, so for now I'm not changing it :) */
	int i = 0;
  //setup first layer, we're going to hold on to the first layer
	Mat2D* first = layerSetup(setup, i);
	printf("layer %i\n", i);
	Print2DMatrix(first);
	printf("layer %i-- rows:%i, cols:%i\n\n", i, first->rows, first->columns);
	++i;

  Mat2D* prev = layerSetup(setup, i); //setup second layer
	printf("layer %i\n", i);
	Print2DMatrix(prev);
	printf("layer %i-- rows:%i, cols:%i\n\n", i, prev->rows, prev->columns);
	++i;

	(*first).next = (Mat2D*)prev; //Link first to 2nd
  Mat2D* next;
	/*Iterate through the layer setup and make the layers*/
	for (; i < setup.layers;) {
    next = layerSetup(setup, i); //setup ith layer
		printf("layer %i\n", i);
		Print2DMatrix(next);
		printf("layer %i-- rows:%i, cols:%i\n\n", i, next->rows, next->columns);
		(*prev).next = (Mat2D*)next; //link i-1'th to ith
		prev = next;
		++i;
	}
	(*next).next = NULL;//give the last layer's next a null pointer
	first->end = next;
	//return the first layer as it will have the links  to all
	return first;
}

Mat2D* sendActual(Mat2D* actual) {
	/*A function to send the training set output values to the GPU for comparison
	*/
  Mat2D* d_act = newMat2D(actual->rows, actual->columns);
	printf("2: %f, 4: %f\n", actual->cells[2], actual->cells[4]);
  d_act->gpuSetup(actual->cells, d_act->Cells);
	return d_act;
}

void nodeRetrieve(Mat2D* &nodes, bool free = true) {
  /*Code for retrieving layer arrays from GPU
  */
  nodes->gpuRetrieve(free);
  Mat2D* next = nodes->next;
  while (next != NULL) {
    next->gpuRetrieve(free);
    next = next->next;
  }
}

void initNodes(Mat2D* &nodes) { //used to return mat2d*
	/*Code for processing the nodes.
	  This should eventually be improved with aSync.The idea here is to try to 
    keep the transfers to the GPU to a minimum Eventually some of this code
    should be put onto the GPU
	  */
	printf("\n\n============= Initializing Nodes ===============\n");

	Mat2D* next = nodes->next;
  nodes->gpuSetup(); // Send first, and get back pointer to device-first
  next->gpuSetup(); // Send next, and get back pointer to device-next
	dim3 tPb(BLKSZ, BLKSZ); // Standard tpb code
  dim3 nb(ceil((double)next->columns / tPb.x), 
          ceil((double)next->rows / tPb.y));
	printf("Layer 1\n");
  LogNodeFunctionKernel << <nb, tPb >> > (*next->dev, *nodes->dev);
	cudaError_t errCode = cudaDeviceSynchronize(); // Sync threads
	printf("GPU Device Synchronization: %s\n", cudaGetErrorString(errCode));

  Mat2D* prev = next;
	next = next->next; // Move through linked list

	int i = 2; // For tracking in the print code
	while (next != NULL) { // Go until end of host linked list
    next->gpuSetup(); // Send next layer
    prev->next = next; // Building device linked list
		// Basically repeated from above
		dim3 nb(ceil((double)next->columns / (double)tPb.x), 
            ceil((double)next->rows / (double)tPb.y));
		printf("Layer %i\n", i);
    LogNodeFunctionKernel <<<nb, tPb >>> (*next->dev, *prev->dev);
		errCode = cudaDeviceSynchronize();
		prev = next;
		next = next->next; //move through linked list
    ++i;
	}
  prev->next = NULL;
  nodes->end = prev;
}

Mat2D* updateNodes(Mat2D* d_nodes, float alpha) {
	/*Code to update the node weights during the learning cycle
	*/

	Mat2D* t = d_nodes;
	printf("==================STARTING NODE UPDATE========================= \n");
	int i = 0;
	//go through linked list of layer arrays
  dim3 tPb(BLKSZ, BLKSZ);
	while (t != NULL) {
		printf("\nLayer %i Update \n", i);
    dim3 nb(ceil((double)t->columns / tPb.x), ceil((double)t->rows / tPb.y));
		UpdateNodesKernel<<<nb, tPb>>>(*t->dev, alpha);
		cudaError_t errCode = cudaDeviceSynchronize();
		printf("\nNode Update: %s\n", cudaGetErrorString(errCode));
		errCode = cudaGetLastError();
		if (errCode != cudaSuccess)
		{
			fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(errCode));
			exit(-1);
		}
		t = t->next;
		++i;
	}
	cudaError_t errCode = cudaDeviceSynchronize();
	printf("\nNode Update: %s\n", cudaGetErrorString(errCode));
	printf("==================NODE UPDATE COMPLETE========================= \n");
	return d_nodes;
}
/////////////////////////////////////////////////////////////////////////////80
/* processNodes-
Main processing function...

Travels through the linked list, calling the main logistic forward function
Reaches the end and pivots, comparing the output with the actual value in the training set
Travels backwards through the recursion return path and calculates the change to the node weights

*/
Mat2D* processNodes(Mat2D* d_n, bool learn = false, Mat2D* actual = NULL,
                    int run = 0, Mat2D* last = NULL) {
	// Travels through the linked list, calling the main logistic forward function
  dim3 tPb(BLKSZ, BLKSZ);
  dim3 dimGrid(ceil((double)d_n->columns / tPb.x), 
               ceil((double)d_n->rows / tPb.y));
	if (d_n->next != NULL) {
		//printf("-Calc Forward-\n");
		LogNodeFunctionKernel <<< dimGrid, tPb >>> (*d_n->next->dev, *d_n->dev);
    cudaError_t errCode = cudaDeviceSynchronize();
		printf("Calc Forward STATUS: %s\n\n", cudaGetErrorString(errCode));
		d_n = processNodes(d_n->next, learn, actual, run, d_n);
	}
	// Reaches the end and pivots, comparing the output with the actual value. 
	else if (learn) {
		//printf("-Calc Pivot-\n");
		LogisticOutputLayerPivotKernel <<< dimGrid, tPb >>> (*last->dev, *d_n->dev,
                                                         *actual->dev, run);
		cudaError_t errCode = cudaDeviceSynchronize();
		printf("Calc Pivot STATUS: %s\n\n", cudaGetErrorString(errCode));
	}
	// Travels backwards through the recursion return path and calculates the 
  // change to the node weights
	if (learn && d_n->next != NULL && last != NULL) {
		//printf("-Calc Backwards-\n");
		//printf("dX:\n");
		LogisticBackpropagationKernel <<< dimGrid, tPb >>> (*last->dev, *d_n->dev,
                                                        *d_n->next->dev);
		cudaError_t errCode = cudaDeviceSynchronize();
		printf("\n Calc Backwards STATUS: %s\n\n", cudaGetErrorString(errCode));
	}
	if (last != NULL) {
		return last;
	}
	else {
		return d_n;
	}
}

float pullBatchErr(Mat2D* d_last, float bSize) {
  printf("Pulling output from last node- ");
  d_last->gpuRetrieve(d_last->X);
  float out = d_last->dX[d_last->columns] / bSize;
	return out;
}

void e1(void) {
  _CrtDumpMemoryLeaks();
}

int main(int argc, char* argv) {
  atexit(e1);
	printf("============= Initializing ===============\n");
	cudaDeviceProp Dev;
	if (cudaGetDeviceProperties(&Dev, 0) == cudaSuccess) {
		printf("................Hardware Properties................\n");
		printf(".....Device Name: %s\n", Dev.name);
		printf(".....Compute Version: %i\n", Dev.major);
		printf(".....Max Clock Rate (MHz): %f\n", 
           (float)Dev.clockRate / (float)(1000 * 1000));
		printf(".....Total Global Memory (MB): %f\n", 
           (float)Dev.totalGlobalMem / (float)(1000*1000));
		printf(".....Total Shared Memory Per Block (kB): %f\n", 
           (float)Dev.sharedMemPerBlock / (float)(1000));
		printf(".....Max Threads Per Block: %i\n", Dev.maxThreadsPerBlock);
		printf(".....Max Thread Dim: %i\n", *Dev.maxThreadsDim);
		printf(".....Max Grid size: %i\n", *Dev.maxGridSize);
		printf(".....# of MultiProcessors: %i\n", Dev.multiProcessorCount);
		printf(".....Max Threads Per MultiProcessor: %i\n", 
           Dev.maxThreadsPerMultiProcessor);
		printf("...................................................\n");
	}
	Config setup("");
	printf("in: %s, act: %s, out: %s\n", setup.in.c_str(), 
         setup.act.c_str(), setup.out.c_str());
	Timer Time(setup.timer);
	
	//Below is a layer setup variable, this takes an array where each value is the number of nodes in that layer
	LaySet lay;
	lay.layers = setup.layers;
	lay.nPl = setup.nodesPerlayer;

	//below is the training set inputs as a vector (turned into a Mat2D)... this is currently just hard-coded.
	//each row is the next input for the traning set
	Mat2D* inputs = CsvToMat2D(setup.in, lay.nPl[0]);

	//below is the training actual values set as a vector... this is currently just hard-coded.
	//each row is the next actual value for the traning set
	Mat2D* actual = CsvToMat2D(setup.act, lay.nPl[lay.layers - 1]);

	int bSize = setup.batchSize; //batch size.... the number of runs before updating the variables
	float alpha = setup.alpha;
	string oString = setup.out; //output file name

  Mat2D* d_in = newMat2D(inputs->rows, inputs->columns);
  d_in->gpuSetup(inputs->cells, d_in->Cells);

	Mat2D* first = hiddenSetup(lay); // setup the hidden layers based on the layset variable lay 
  initNodes(first); // initial run through with all inputs set to one
  Mat2D* d_first = first;
  Mat2D* d_act = sendActual(actual); // send the training set outputs to the GPU

	ofstream outFile;
	outFile.open(oString);
	outFile << "Batch #, ";
	for (int i = 0; i < d_act->columns; ++i) {
		outFile << "Output Node " << i << " error squared, ";
	}
	// Now we run through all of the training set... This will later be replaced
  // by other options as to when the learning stops.
	float bErr;
	printf("\n\n========================================================== Begin Batch Run =====================================================\n");
	printf("---Batch Size: %i\n", bSize);
  dim3 tPb(BLKSZ, BLKSZ);
  dim3 nb(ceil((double)d_first->rows / tPb.y), 
          ceil((double)d_first->columns / tPb.x));
	for (int cyc = 0; cyc < 1; ++cyc) {
		int rn = 0;
		for (; rn < inputs->columns * inputs->rows;) {
			int b = bSize;
			while (b > 0) { 
        // This loops until the batch size is met and moves to the update step.
				printf("\n--Run: %i\n", rn + b*cyc);
				ChangeInputsKernel <<< nb, tPb >>> (*d_first->dev, *d_in->dev, rn);
        cudaError_t errCode = cudaDeviceSynchronize();
				printf("Calc Update Node: %s\n\n", cudaGetErrorString(errCode));
				d_first = processNodes(d_first, true, d_act, rn); //process the nodes
				b = b - 1; //next batch index
				rn = rn + 1; //next run index
			}
			bErr = pullBatchErr(d_first->end, (float)bSize);
			int c = actual->columns;
			outFile << endl << (rn / inputs->columns)/bSize;
			for (int i = 0; i < c; ++i) {
				outFile << ", " << bErr;
			}
			d_first = updateNodes(d_first, alpha); //update the node weights 
		}
	}
	printf("\n============================================================= End Batch Run ======================================================\n");
  cudaError_t errCode = cudaGetLastError();
	printf("Starting nodeRetrieve, last CUDA error: %s\n",
         cudaGetErrorString(errCode));
	//first = nodeRetrieve(d_first, first);
  nodeRetrieve(first);
	Mat2D* temp;
	int i = 0;
	while (first != NULL) {
		printf("Layer %i\n", i);
		Print2DMatrix(first);
		temp = (Mat2D*)first;
		first = first->next;
		//free(temp);
    delete temp;
		++i;
	}
	cudaFree(d_in);
	free(d_in);
	//outFile->close;
//++++++++!!!!!!!! If having errors on external code, the reset below may cause it================
	//errCode = cudaDeviceReset(); //clear any remaining items on device...
	printf("GPU reset: %s\n", cudaGetErrorString(errCode));
	return 0;
	}


/*Austin's useful debugging tools
=======================================================================
This one is good just to put in the code for status:

cudaError_t errCode = ...
printf("Retrieving nodes from GPU: %s\n", cudaGetErrorString(errCode));

========================================================================
This one is good to put after kernal execution to get any errors w/in the kernal:
Put it after a thread sync call

cudaError_t errCode = cudaGetLastError();
if (errCode != cudaSuccess)
{
fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(errCode));
exit(-1);
}


========================================================================
Below is some good console debug code for errors within the kernal:

nvcc -lineinfo -o matops matops.cu
cuda-memcheck ./matops |more

========================================================================
Below is a good way of deciphering error messages:
www.google.com
*/