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
v0: 1/15/2017- original
v0.01: 1/15/2017- added transpose

===========================================
*/
//required headers:
#include "nodeSet.h" //v0; includes stdio.h
//optional headers:
#include <stdlib.h> //I'm here so I don't get fined... aka so malloc doesn't show as error... will compile without


__global__ void mAddKernel2D(Mat2D d_a, Mat2D d_b, Mat2D d_o) {
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (r > d_o.rows || c > d_o.columns) return;

	d_o.cells[r * d_o.columns + c] = d_a.cells[r * d_o.columns + c] + d_b.cells[r * d_o.columns + c];
}

__global__ void mSubKernel2D(Mat2D d_a, Mat2D d_b, Mat2D d_o) {
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (r > d_o.rows || c > d_o.columns) return;

	d_o.cells[r * d_o.columns + c] = d_a.cells[r * d_o.columns + c] - d_b.cells[r * d_o.columns + c];
}

__global__ void mTransKernel2D(Mat2D d_i, Mat2D d_o) {

	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	//if out-of-bounds stop
	if (r > d_o.rows || c > d_o.columns) return;

	d_o.cells[r * d_o.columns + c] = d_i.cells[c * d_i.columns + r];
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

Mat2D cudaMAdd2D(Mat2D f_mA, Mat2D f_mB) {
	// ADD ERROR HANDLING
	//Send input matricies to GPU and return d_mA so GPU memory can be deallocated later
	printf("---------Addition---------\n");
	Mat2D d_mA = cudaMSend2D(f_mA, true, "matrix A");
	Mat2D d_mB = cudaMSend2D(f_mB, true, "matrix B");

	//Create output matrix and allocate memory on GPU. Returns d_out to access result/deallocate mem
	Mat2D out;
	out.rows = d_mA.rows;
	out.columns = d_mA.columns;
	out.cells = (float*)malloc(out.rows * out.columns * sizeof(float));
	Mat2D d_out = cudaMSend2D(out, false, "Output matrix");

	//setup CUDA architecture and run kernel
	dim3 threadsPerBlock(16, 16); //each block will contain 16 by 16 threads
	dim3 numBlocks((d_out.columns + threadsPerBlock.x - 1) / threadsPerBlock.x, //number of blocks on x dimension of grid
		(d_out.rows + threadsPerBlock.y) / threadsPerBlock.y); //number of blocks on y dimension of grid
	mAddKernel2D <<<numBlocks, threadsPerBlock >> > (d_mA, d_mB, d_out); //run's kernal
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

	//print result
	print2DMat(out, "\n\n--Addition Results--\nOutput ");
	printf("--------------------------------\n");

	//deallocate GPU memory
	cudaFree(d_mA.cells);
	cudaFree(d_mB.cells);
	cudaFree(d_out.cells);

	return out;
	//~~ALB
}

Mat2D cudaMTrans2D(Mat2D f_m) {
	printf("--------- Transposition ---------\n");
	Mat2D d_m = cudaMSend2D(f_m, true, "Original");

	Mat2D out;
	out.rows = f_m.columns;
	out.columns = f_m.rows;
	out.cells = (float*)malloc(out.rows*out.columns * sizeof(float));
	Mat2D d_o = cudaMSend2D(out, true, "output");

	dim3 tPBlock(16, 16);
	dim3 nBlocks((out.columns + tPBlock.x - 1) / tPBlock.x, (out.rows + tPBlock.y - 1) / tPBlock.y);
	mTransKernel2D << <nBlocks, tPBlock >> > (d_m, d_o);
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
	errCode = cudaMemcpy(out.cells, d_o.cells, d_o.rows * d_o.columns * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Pulling Output matrix from GPU: %s\n", cudaGetErrorString(errCode));

	//print result
	print2DMat(out, "\n\n--Transposition Results--\nOutput ");
	printf("--------------------------------\n");

	cudaFree(d_m.cells);
	cudaFree(d_o.cells);

	return out;
	/*int rows;
	int cols;

	rows = f_m.rows / par;
	cols = f_m.columns / par;*/

}

Mat2D cudaMMult2D(Mat2D f_mA, Mat2D f_mB) { /*2D Matrix multiplication algorithm
											needs: mMultKernel2D<<<nBlocks, tPb>>>(Mat2D d_mA, Mat2D d_mB, Mat2D d_out),
											cudaMSend2D(Mat2D f_mX, Bool TF, const char* ID), nodeSet.h
											*/

											//check dimensions, err if incorrect
	if (f_mA.columns != f_mB.rows) {
		printf("ERROR: Incorrect array dimensions A*B. Number of columns in A must equal number of rows in B.\n");
		printf("A: %i x %i, B: %i x %i\n", f_mA.rows, f_mA.columns, f_mB.rows, f_mB.columns);
		Mat2D err;
		err.rows = -1;
		return err;
	}

	//Send input matricies to GPU and return d_mA so GPU memory can be deallocated later
	printf("---------Multiplication---------\n");
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
	mMultKernel2D << <numBlocks, threadsPerBlock >> > (d_mA, d_mB, d_out); //run's kernal
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

	//print result
	print2DMat(out, "\n\n--Multiplication Results--\nOutput ");
	printf("--------------------------------\n");

	//deallocate GPU memory
	cudaFree(d_mA.cells);
	cudaFree(d_mB.cells);
	cudaFree(d_out.cells);

	return out;
	//~~ALB
}

//Mat2D cudaSSE2d(Mat2D f_y, Mat2D f_h) {
//	Mat2D d_y = cudaMSend2D(f_y, true);
//	Mat2D d_f = cudaMSend2D(f_h, true);
//	Mat2D out;
//	
//}

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

	printf("----------- Inputs -----------\n");

	//Check for bad inputs
	if (sizeof(a) / sizeof(float) != rA*cA) {
		printf("Number of elements in A does not equal rowA*colA\n");
		return 0;
	}
	if (sizeof(b) / sizeof(float) != rB*cB) {
		printf("Number of elements in B does not equal rowB*colB\n");
		return 0;
	}

	//turn input vectors into matricies
	Mat2D aM = vecToMat2D(a, rA, cA);
	print2DMat(aM, "\nA ");
	Mat2D bM = vecToMat2D(b, rB, cB);
	print2DMat(bM, "\nB ");

	
	//call multiplication algorithm. Returns resulting matrix
	Mat2D cM = cudaMMult2D(aM, bM);
	if (cM.rows == -1) {
		//dealocate memory from other function calls
		free(aM.cells);
		free(bM.cells);
		return 0;
	}
	free(cM.cells);
	
	cM = cudaMTrans2D(aM);
	free(cM.cells);
	cM = cudaMTrans2D(bM);
	free(cM.cells);
	cM = cudaMAdd2D(aM, bM);
	//dealocate memory from other function calls
	free(aM.cells);
	free(bM.cells);
	free(cM.cells);


	return 0;
	}
