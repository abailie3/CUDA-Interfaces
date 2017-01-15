/*
======== CUDA matrix operations v0 ========
			By: Austin Bailie

Various matrix operations computed on the GPU.
Over-commented for educational purposes

Adapted from:
	-nVidia's CUDA Programming guide
	-Robert Hochberg (1/24/16): http://bit.ly/2iA8jDc
===========================================
*/
/*
============ Change Log ===================
v0: 1/15/2017
	-original

===========================================
*/
//required headers:
#include <string.h>
#include "nodeSet.h" //v0; includes stdio.h
//optional headers:
#include <stdlib.h> //I'm here so I don't get fined... aka so malloc doesn't show as error... will compile without
#include "cuda_runtime.h" //same deal as stdlib.h's comment
#include "device_launch_parameters.h" //same deal as stdlib.h's comment


void print2DMat(Mat2D out, const char* prompt = "") { /*simple method to print matrix
	needs: nodeSet.h, string.h*/
	printf("%sMatrix Values:\n{\n", prompt); //just making it pretty
	for (int i = 0; i < out.rows; ++i) { //iterate through each row/col and print
		printf("    "); //again, making pretty
		for (int t = 0; t < out.columns; ++t)
			printf("%f, ", out.cells[i*out.columns + t]);
		printf("\n");
	}
	printf("}\n");
	//~~ALB
}

Mat2D vecToMat2D(float f_vector[], int f_rows, int f_cols) { /*convert vector to a mat2D
	needs: nodeSet.h, print2DMat(Mat2D out)
	I don't really need stdlib.h, but malloc shows a pesky error if not... will compile without
	*/
	//output matrix
	Mat2D out;
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

Mat2D cudaMSend2D(Mat2D iM, bool copy, const char* iD) { /*Handles GPU memory allocaion/memory transfer to GPU.
	needs: nodeset.h
	copy boolean determines if the matrix values should be copied into the allocated memory on GPU
	iD takes a constant char pointer of the matrix name/ID
	*/
	
	//device's copy of the input matrix
	Mat2D d_M; 
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
}

__global__ void mMultKernel2D(Mat2D d_a, Mat2D d_b, Mat2D d_c) { /*CUDA kernal for 2D matrix multiplication
																 needs: nodeSet.h,
																 I don't really need cuda_runtime.h or device_launch parameters, but they prevent errors... will compile without
																 */

	float oVal = 0;//output value

	int r = blockIdx.y * blockDim.y + threadIdx.y; //getting row based on CUDA thread/block index
	int c = blockIdx.x * blockDim.x + threadIdx.x; //getting column based on CUDA thread/block index

												   //if out-of-bounds stop
	if (r > d_c.rows || c > d_c.columns) return;

	//add up each A(r,i)*B(i,c) into oVal
	for (int i = 0; i < d_a.columns; ++i)
		oVal += d_a.cells[r * d_a.columns + i] * d_b.cells[i*d_b.columns + c];

	//assign the oVal to the output matrix
	d_c.cells[r * d_c.columns + c] = oVal;
	//~~ALB
}

Mat2D cudaMMult2D(Mat2D f_mA, Mat2D f_mB) { /*2D Matrix multiplication algorithm
	needs: mMultKernel2D<<<nBlocks, tPb>>>(Mat2D d_mA, Mat2D d_mB, Mat2D d_out),
		   cudaMSend2D(Mat2D f_mX, Bool TF, const char* ID), nodeSet.h
	*/
	//Send input matricies to GPU and return d_mA so GPU memory can be deallocated later
	Mat2D d_mA = cudaMSend2D(f_mA, true, "matrix A");
	Mat2D d_mB = cudaMSend2D(f_mB, true, "matrix B");
	
	//Create output matrix and allocate memory on GPU. Returns d_out to access result/deallocate mem
	Mat2D out;
	out.rows = d_mA.rows;
	out.columns = d_mB.columns;
	out.cells = (float*)malloc(out.rows * out.columns * sizeof(float));
	Mat2D d_out = cudaMSend2D(out, false, "Output matrix");

	//setup CUDA architecture and run kernel
	dim3 threadsPerBlock(16, 16); //each block will contain 16 by 16 threads
	dim3 numBlocks((d_out.columns + threadsPerBlock.x - 1) / threadsPerBlock.x, //number of blocks on x dimension of grid
		            (d_out.rows + threadsPerBlock.y) / threadsPerBlock.y); //number of blocks on y dimension of grid
	mMultKernel2D <<<numBlocks, threadsPerBlock >>> (d_mA, d_mB, d_out); //run's kernal
	cudaError_t errCode = cudaThreadSynchronize(); //synchronize cores to ensure everthing has been run
	printf("GPU Thread Synchronization: %s\n", cudaGetErrorString(errCode));
	
	//debug code to find errors in execution of kernel
	errCode = cudaGetLastError();
	if (errCode != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(errCode));
		exit(-1);
	}

	//retrieve output matrix from GPU memory
	errCode = cudaMemcpy(out.cells, d_out.cells, d_out.rows * d_out.columns * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Pulling Output matrix from GPU: %s\n", cudaGetErrorString(errCode));
	
	
	//deallocate GPU memory
	cudaFree(d_mA.cells);
	cudaFree(d_mB.cells);
	cudaFree(d_out.cells);

	return out;
	//~~ALB
}



int main(int argc, char* argv) {
	float a[] = {
		2, 4, 6, 7,
		1, 3, 4, 6
	};
	int rA = 2;
	int cA = 4;

	float b[] = {
		0, 4,
	    2, 3, 
		2, 4,
		4, 4
	};
	int rB = 4;
	int cB = 2;

	//check dimensions, err if incorrect
	if (cA != rB) {
		printf("ERROR: Incorrect array dimensions A*B. Number of columns in A must equal number of rows in B.\n");
		printf("A: %i x %i, B: %i x %i\n", rA, cA, rB, cB);
		return 0;
	} 

	//turn input vectors into matricies
	Mat2D aM = vecToMat2D(a, rA, cA);
	print2DMat(aM, "A ");
	Mat2D bM = vecToMat2D(b, rB, cB);
	print2DMat(bM, "B ");

	//call multiplication algorithm. Returns resulting matrix
	Mat2D cM = cudaMMult2D(aM, bM);

	//print result
	print2DMat(cM, "________________________\nResulting ");
	
	//dealocate memory from other function calls
	free(aM.cells);
	free(bM.cells);
	free(cM.cells);

	return 0;
	}
