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
v0: 1/15/2017		- original

v0.01: 1/15/2017	- added transpose

v0.1: 1/21/2016		- added various matrix math functions
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

===========================================
*/
//required headers:
#include "nodeSet.h" //v0; includes stdio.h and now math.h
//optional headers:
//#include <stdlib.h> //so malloc doesn't show as error... will compile without

using namespace std;
#define BLKSZ 16

__global__ void mAddKernel2D(Mat2D d_a, Mat2D d_b, Mat2D d_o) {
	/*CUDA kernel for 2d matrix addition
	  needs: nodeSet.h,
	  I don't really need cuda_runtime.h or device_launch parameters, but they prevent errors... will compile without
	*/
	int r = blockDim.y * blockIdx.y + threadIdx.y; //getting row based on CUDA thread/block index
	int c = blockDim.x * blockIdx.x + threadIdx.x; //getting column based on CUDA thread/block index
	
	//if out-of-bounds stop
	if (r > d_o.rows || c > d_o.columns) return;

	//add
	d_o.cells[r * d_o.columns + c] = d_a.cells[r * d_o.columns + c] + d_b.cells[r * d_o.columns + c];
	//~~ALB
}

__global__ void mSubKernel2D(Mat2D d_a, Mat2D d_b, Mat2D d_o) {
	/*CUDA kernel for 2d matrix subtraction
	  needs: nodeSet.h,
	  I don't really need cuda_runtime.h or device_launch parameters, but they prevent errors... will compile without
	*/
	int r = blockDim.y * blockIdx.y + threadIdx.y; //getting row based on CUDA thread/block index
	int c = blockDim.x * blockIdx.x + threadIdx.x; //getting column based on CUDA thread/block index

	//if out-of-bounds stop
	if (r > d_o.rows || c > d_o.columns) return;

	//subtract
	d_o.cells[r * d_o.columns + c] = d_a.cells[r * d_o.columns + c] - d_b.cells[r * d_o.columns + c];
}

__global__ void mTransKernel2D(Mat2D d_i, Mat2D d_o) {
	/*CUDA kernel for 2d matrix subtraction
	  needs: nodeSet.h
	  I don't really need cuda_runtime.h or device_launch parameters, but they prevent errors... will compile without
	*/
	int r = blockIdx.y * blockDim.y + threadIdx.y; //getting row based on CUDA thread/block index
	int c = blockIdx.x * blockDim.x + threadIdx.x; //getting column based on CUDA thread/block index

	//if out-of-bounds stop
	if (r > d_o.rows || c > d_o.columns) return;
	
	//switch rows and columns
	d_o.cells[r * d_o.columns + c] = d_i.cells[c * d_i.columns + r];
}

__global__ void mMultKernel2D(Mat2D d_a, Mat2D d_b, Mat2D d_c) { 
	/*CUDA kernel for 2D matrix multiplication
	  needs: nodeSet.h,
	  I don't really need cuda_runtime.h or device_launch parameters, but they prevent errors... will compile without
	  
	  Adapted from:
	  -Robert Hochberg (1/24/16): http://bit.ly/2iA8jDc
	  */

	float oVal = 0;//output value

	int r = blockIdx.y * blockDim.y + threadIdx.y; //getting row based on CUDA thread/block index
	int c = blockIdx.x * blockDim.x + threadIdx.x; //getting column based on CUDA thread/block index

    //if out-of-bounds stop
	//if (r > d_c.rows || c > d_c.columns) return; So I changed my index out of bounds protection, see below

	//add up each A(r,i)*B(i,c) into oVal
	if (r < d_c.rows && c < d_c.columns) {
		for (int i = 0; i < d_a.columns; ++i)
			oVal += d_a.cells[r * d_a.columns + i] * d_b.cells[i*d_b.columns + c];
	}
	else {
		return;
	}
	//assign the oVal to the output matrix
	d_c.cells[r * d_c.columns + c] = oVal;
	//~~ALB
}

__device__ void logistic2DD(Mat2D d_layer, Mat2D d_prev, int r, int c, int lr, int lc) {
	/*CUDA kernel for logistic nodes, this is the main forward (input to output) function

	this function is going forward on the recursion path of the host function
	needs: nodeset.h, math.h
	*/
	//int r = blockDim.y * blockIdx.y + threadIdx.y; //getting row based on CUDA thread/block index
	//int c = blockDim.x * blockIdx.x + threadIdx.x; //getting column based on CUDA thread/block index

	//if (r < 1 && c < lc) { //we are only updating the first layer (node outputs) of the array
	float l = d_layer.cells[lc + c];//add in the bias

													/*we need to add up all of x*w of the previous layer, where:
													x is the output of a node in the previous layer, stored in row 1 of the previous layer
													w is the weight of the corresponding output, stored in column c of the current layer*/
	for (int i = 2; i < lr;) { //start at 2 since 0 has the outputs and 1 has the bias

		l += d_layer.cells[i * lc + c] * d_prev.cells[i - 2];

		//printf("%i,%i>>l: %f, x: %f, w: %f\n", i -2, c, l, d_prev.cells[i - 2], d_layer.cells[i * d_layer.columns + c]);
		i++;
	}

	d_layer.cells[c] = 1 / (1 + exp(-l)); //now we calculate the output of this node
	printf("Row: %i, Col: %i, Input: %f, Value: %f \n", r, c, l, d_layer.cells[c]);

	/*}
	else
	{
		return;
	}*/
	//~~ALB
}

__global__ void logistic2D(Mat2D d_layer, Mat2D d_prev) {
	/*CUDA kernel for logistic nodes, this is the main forward (input to output) function

	  this function is going forward on the recursion path of the host function
	  needs: nodeset.h, math.h
	  */
	int r = blockDim.y * blockIdx.y + threadIdx.y; //getting row based on CUDA thread/block index
	int c = blockDim.x * blockIdx.x + threadIdx.x; //getting column based on CUDA thread/block index
	//printf("blockIdx.x %i, blockIdx.y %i\n", blockIdx.x, blockIdx.y);
	int lr = d_layer.rows;
	int lc = d_layer.columns;
	//printf("r %i, lr %i, c %i, lc %i, bIx %i, bIy %i, tIx %i, tIy %i\n", r, lr, c, lc, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
	if (r < 1 && c < lc) {
		logistic2DD(d_layer, d_prev, r, c, lr, lc);
	}
	else {
		return;
	}
	//if (r < 1 && c < d_layer.columns) { //we are only updating the first layer (node outputs) of the array
	//	float l = d_layer.cells[d_layer.columns + c];//add in the bias

	//	/*we need to add up all of x*w of the previous layer, where:
	//			x is the output of a node in the previous layer, stored in row 1 of the previous layer
	//			w is the weight of the corresponding output, stored in column c of the current layer*/
	//	for (int i = 2; i < d_layer.rows;) { //start at 2 since 0 has the outputs and 1 has the bias

	//		l += d_layer.cells[i * d_layer.columns + c] * d_prev.cells[i - 2];
	//		
	//		//printf("%i,%i>>l: %f, x: %f, w: %f\n", i -2, c, l, d_prev.cells[i - 2], d_layer.cells[i * d_layer.columns + c]);
	//		i++;
	//	}

	//	d_layer.cells[c] = 1 / (1 + exp(-l)); //now we calculate the output of this node
	//	printf("Row: %i, Col: %i, Input: %f, Value: %f \n", r, c, l, d_layer.cells[c]);

	//}
	//else
	//{
	//	return;
	//}
	//~~ALB
}

__device__ float rowSumDX(Mat2D d_a, int r) {
	float out = 0;
	int cc = d_a.columns;
	for (int i = 0; i < cc; ++i) {
		out += d_a.dX[r * cc + i];
	}
	return out;
}

__device__ void nodeBackwardLogD(Mat2D d_prev, Mat2D d_cur, Mat2D d_next, int r, int c, int cr, int cc) {
	/*CUDA kernel for back propagation on logistic nodes
	the .dTh matrix is a matrix of the change in weights for the corresponding node-input pair
	the .dX matrix is a matrix of the change in the cascaded error term for the corresponding node-input pair
	the first row of .dX holds the sum  of the .dX terms in the next layer of the corresponding input

	This code is going backward (along the return path of the host recursive function)
	needs: nodeset.h, math.h
	*/
	//int r = blockIdx.y * blockDim.y + threadIdx.y;
	//int c = blockIdx.x * blockDim.x + threadIdx.x;
	//int cr = d_current.rows;
	//int cc = d_current.columns;

	if (r > 0 && r < cr && c < cc) { //only operating on the weights and bias
		//printf("%f", d_current.dX[c]); //debug code
		
		float tX = d_cur.cells[c]; //initialize variable for the dX/dL term
		float bX;
		bX = tX - (tX * tX); //the (x- x^2) term of dX/dL
		
		//printf("If 1 True\nR: %i, C: %i\n", r, c); //debug code
		//printf("d_next.columns %i\n", d_next.columns); //debug code

		float DX = d_cur.dX[c]; //the total dX/dx of the next layer is sumed into this layer in the first row of the .dX matrix
		//printf(" dbX_n %f ", dbX_n); //debug code

		if (r == 1) { //change to bias
			float dB = d_cur.dTh[r * cc + c];
			dB = dB + bX * DX;
			d_cur.dTh[r * cc + c] = dB;
			//d_current.dX[d_current.columns + c] = dbX_n;
		}
		else { //change to the weights (theta)
			float dTh = d_cur.dTh[r * cc + c];
			float pX = d_prev.cells[r-2];
			dTh += DX * bX * pX;
			d_cur.dTh[r * cc + c] = dTh;
			d_cur.dX[r * cc + c] = d_cur.dX[c] * bX * d_cur.cells[r * cc + c];

			//d_prev.dX[r - 2] = d_prev.dX[r - 2] + DX*bX*d_current.cells[r * d_current.columns + c]; //sum the dX/dx layer for the previous layer... this needs to go after syncthreads
		}
	}
	else {
		return;
	}
	//~~ALB
}

__global__ void nodeBackwardLog(Mat2D d_prev, Mat2D d_cur, Mat2D d_next) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int cr = d_cur.rows;
	int cc = d_cur.columns;
	nodeBackwardLogD(d_prev, d_cur, d_next, r, c, cr, cc);
	__syncthreads();

	if (r > 1 && r - 2 < d_prev.columns && c == 0) {
		float p = rowSumDX(d_cur, r);
		d_prev.dX[r - 2] = d_prev.dX[r - 2] + p;
	}
	else {
		return;
	}

}

__device__ void outPivotLogD(Mat2D d_prev, Mat2D d_cur, Mat2D actual, int rn, int r, int c, int cr, int cc) {
	/*CUDA kernel for error comparison on logistic nodes
	the .dTh matrix is a matrix of the change in weights for the corresponding node-input pair
	the .dX matrix is a matrix of the change in the cascaded error term for the corresponding node-input pair
	the first row of .dX holds the sum  of the .dX terms in the next layer of the corresponding input

	This code is the pivot of the recursive function (aka from going forward to going backward)
	needs: nodeset.h, math.h
	*/
	
	//int r = blockIdx.y * blockDim.y + threadIdx.y;
	//int c = blockIdx.x * blockDim.x + threadIdx.x;
	//int cr = d_cur.rows;
	//int cc = d_cur.columns;

	if (r > 0 && r < cr && c < cc) {
		//printf("If 1 True\nR: %i, C: %i\n", r, c); //debug code
		//printf("d_cur.columns %i\n", d_cur.columns); //debug code
		float bX = d_cur.cells[c];
		bX = bX - (bX * bX);
		//printf("piv c: %i, run: %i, actual: %f\n", c, rn, actual.cells[rn * actual.columns + c]); //debug code
		float act = actual.cells[rn * actual.columns + c];
		float err = d_cur.cells[c] - act;
		printf("Error Squared: %f\n, actual: %f\n", err * err, act);
		//printf("pre if 2\n");
		
		if (r == 1) { //change to bias
			//printf("True-2\n"); //debug code
			//printf("d_cur.columns %i", d_cur.columns); //debug code
			float dB = d_cur.dTh[r * cc + c];
			dB = dB + bX * err;
			if (c == 0) {
				d_cur.dX[cc] += err*err;
			}
			//d_cur.dTh[r *cc + c] = dB;
			//d_cur.dX[cc + c] = bX * err;
			//printf("dTh(%i, %i): %f, dX: %f\n", r, c, d_cur.dTh[r * d_cur.columns + c], d_cur.dX[d_cur.columns + c]); //debug code
		}
		else { //change to the weights (theta)
			//printf("Else-2\n"); //debug code
			float dTh = d_cur.dTh[r * cc + c];
			float pX = d_prev.cells[r-2];
			dTh += err * bX * pX;
			d_cur.dTh[r * cc + c] = dTh;
			d_cur.dX[r *cc + c] = err * bX * d_cur.cells[r * cc + c];

			//d_prev.dX[r - 2] = d_prev.dX[r - 2] + d_cur.dX[r * d_cur.columns + c]; //sum the dX/dx layer for the previous layer //needs to be in sync threads
		}
	}
	else
	{
		return;
	}
	//~~ALB
}

__global__ void outPivotLog(Mat2D d_prev, Mat2D d_cur, Mat2D actual, int rn) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int cr = d_cur.rows;
	int cc = d_cur.columns;
	outPivotLogD(d_prev, d_cur, actual, rn, r, c, cr, cc);
	__syncthreads();

	if (r > 1 && r - 2 < d_prev.columns && c == 0) {
		float p = rowSumDX(d_cur, r);
		d_prev.dX[r - 2] = d_prev.dX[r - 2] + p;
	}
	else {
		return;
	}

}

__global__ void updateNodes(Mat2D d_nodes, float alpha) {
	/*CUDA kernel for weight/bias updates
	  This node is the update step of the learning process
	*/
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

	//~~ALB
}

__global__ void changeInK(Mat2D d_nodes, Mat2D d_in, int rn) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (r == 0 && c < d_nodes.columns) {
		printf("Layer 0: R: %i, C: %i, in: %f \n", r, c, d_in.cells[rn * d_in.columns + c]);
		d_nodes.cells[c] = d_in.cells[rn * d_in.columns + c];
	}
	else {
		return;
	}
}

Mat2D cudaMAdd2D(Mat2D f_mA, Mat2D f_mB) {
	if (f_mA.columns != f_mB.columns || f_mA.rows != f_mB.rows) {
		printf("ERROR: Incorrect array dimensions A + B. Sizes must be equal.\n");
		printf("A: %i x %i, B: %i x %i\n", f_mA.rows, f_mA.columns, f_mB.rows, f_mB.columns);
		Mat2D err;
		err.rows = -1;
		return err;
	}
	//Send input matricies to GPU and return d_mA so GPU memory can be deallocated later
	printf("--------- Addition ---------\n");
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
		(d_out.rows + threadsPerBlock.y - 1) / threadsPerBlock.y); //number of blocks on y dimension of grid
	mAddKernel2D << <numBlocks, threadsPerBlock >> > (d_mA, d_mB, d_out); //run's kernal
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

Mat2D cudaMSub2D(Mat2D f_mA, Mat2D f_mB) {
	if (f_mA.columns != f_mB.columns || f_mA.rows != f_mB.rows) {
		printf("ERROR: Incorrect array dimensions A + B. Sizes must be equal.\n");
		printf("A: %i x %i, B: %i x %i\n", f_mA.rows, f_mA.columns, f_mB.rows, f_mB.columns);
		Mat2D err;
		err.rows = -1;
		return err;
	}
	//Send input matricies to GPU and return d_mA so GPU memory can be deallocated later
	printf("--------- Subtraction ---------\n");
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
		(d_out.rows + threadsPerBlock.y - 1) / threadsPerBlock.y); //number of blocks on y dimension of grid
	mSubKernel2D <<<numBlocks, threadsPerBlock >>> (d_mA, d_mB, d_out); //run's kernal
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
	print2DMat(out, "\n\n--Subtraction Results--\nOutput ");
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
	//out.id = getID(mIds, 1);
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


	//~~ALB
}

Mat2D cudaMMult2D(Mat2D f_mA, Mat2D f_mB) { 
	/*2D Matrix multiplication algorithm
	  needs: mMultKernel2D<<<nBlocks, tPb>>>(Mat2D d_mA, Mat2D d_mB, Mat2D d_out),
	  cudaMSend2D(Mat2D f_mX, Bool TF, const char* ID), nodeSet.h

	  Adapted from:
	  -Robert Hochberg (1/24/16): http://bit.ly/2iA8jDc
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
	//out.id = getID(master, 1);
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

Mat2D layerSetup(laySet setup, int indx, bool onesFirst = true) { 
	/*Setup of the node layers:
	  Each array has the output node in row zero, and the weights for each input in the rows below.
	  So, # of rows = # of nodes from previous layer + 2 (the output node row and a bias row on row 1)
	  This also means, # of rows = # of columns of previous layer + 2
	*/
	Mat2D out;
	out.columns = setup.nPl[indx]; 
	out.rows = 2;
	/* The first layer will have no weight or bias, but has to have 2 rows	*/
	if (indx == setup.layers - 1) {
		out.rows = setup.nPl[indx - 1] + 2;
	}
	else if (indx != 0)
	{	
		out.rows = setup.nPl[indx - 1] + 2;
	}

	//allocate memory for cells
	out.cells = (float*)malloc(out.rows * out.columns * sizeof(float));
	out.dTh = (float*)malloc(out.rows * out.columns * sizeof(float));
	out.dX = (float*)malloc(out.rows * out.columns * sizeof(float));
	//initialize array with 1's in row 0 and zeros in rest
	float zer = 0;
	float bz = 1;
	for (int r = 0; r < out.rows; ++r) {
		//float a = 1 / (((float)indx + 1)*((float)r + 1));
		srand(r + 2 + time(NULL));
		for (int c = 0; c < out.columns; ++c) {
			out.cells[r * out.columns + c] = (float)pow((2.01 - (rand()%400)/100.1)/out.rows ,zer)*bz;
			out.dTh[r * out.columns + c] = 0;
			out.dX[r * out.columns + c] = 0;
		}
		bz = 1;
		if (r == 0) bz = 0;
		zer = 1;
	}

	return out;
	//~~ALB
}

Mat2D* hiddenSetup(laySet setup) {
	/*Setup of the 'hidden' layers and input layer
	  The goal is to setup a linked list. Its the dream..
	  From here on we're passing pointers rather than the structure
	*/
	printf("\n\n============= Hidden Layers Setup ===============\n");
	Mat2D *first;
	Mat2D *prev;
	Mat2D *next;
	
	/*The below code is my way of setting up a linked list. 
	  I think there's a better way to code this without dereferencing so much,
	  but it works, so for now I'm not changing it :) */
	int i = 0;

	first = (Mat2D*)malloc(sizeof(Mat2D));//allocate memory
	*first = layerSetup(setup, i);//setup first layer, we're going to hold on to the first layer
	printf("layer %i\n", i);
	print2DMat(*first);
	printf("layer %i-- rows:%i, cols:%i\n\n", i, first->rows, first->columns);
	i++;

	prev = (Mat2D*)malloc(sizeof(Mat2D));//allocate memory
	*prev = layerSetup(setup, i); //setup second layer
	printf("layer %i\n", i);
	print2DMat(*prev);
	printf("layer %i-- rows:%i, cols:%i\n\n", i, prev->rows, prev->columns);
	i++;

	(*first).next = (struct Mat2D*)prev; //Link first to 2nd
	
	/*Iterate through the layer setup and make the layers*/
	for (; i < setup.layers;) {
		next = (Mat2D*)malloc(sizeof(Mat2D)); //allocate memory
		*next = layerSetup(setup, i); //setup ith layer
		printf("layer %i\n", i);
		print2DMat(*next);
		printf("layer %i-- rows:%i, cols:%i\n\n", i, next->rows, next->columns);
		(*prev).next = (struct Mat2D*)next; //link i-1'th to ith
		prev = next;
		i++;
	}
	(*next).next = NULL;//give the last layer's next a null pointer
	first->end = next;
	//return the first layer as it will have the links  to all
	return first;

	//~~ALB
}

Mat2D* lmatSend2D(Mat2D* nodes) {
	/*Code for sending the linked matricies to the GPU.
	  We cant use the CudaMsend2D since we want to use pointers.
	  This needs to eventually be moved to the header I think, but for now it stays.
	  */

	Mat2D* d_nodes = (Mat2D*)malloc(sizeof(Mat2D));//This time we have to allocate memory for the device pointer

	cudaError_t errCode = cudaMalloc(&d_nodes->cells, nodes->rows * nodes->columns * sizeof(float)); //allocate mem on GPU
	printf("GPU cudaMalloc Nodes: %s\n", cudaGetErrorString(errCode));
	errCode = cudaMalloc(&d_nodes->dTh, nodes->rows * nodes->columns * sizeof(float)); //allocate mem on GPU
	printf("GPU cudaMalloc dTh: %s\n", cudaGetErrorString(errCode));
	errCode = cudaMalloc(&d_nodes->dX, nodes->rows * nodes->columns * sizeof(float)); //allocate mem on GPU
	printf("GPU cudaMalloc dX: %s\n", cudaGetErrorString(errCode));
	d_nodes->rows = nodes->rows;
	d_nodes->columns = nodes->columns;
	
	
	errCode = cudaMemcpy(d_nodes->cells, nodes->cells, d_nodes->rows * d_nodes->columns * sizeof(float), cudaMemcpyHostToDevice);//copy cell values to GPU
	printf("Memcpy Nodes: %s\n", cudaGetErrorString(errCode));
	errCode = cudaMemcpy(d_nodes->dTh, nodes->dTh, d_nodes->rows * d_nodes->columns * sizeof(float), cudaMemcpyHostToDevice);//copy cell values to GPU
	printf("Memcpy dTh: %s\n", cudaGetErrorString(errCode));
	errCode = cudaMemcpy(d_nodes->dX, nodes->dX, d_nodes->rows * d_nodes->columns * sizeof(float), cudaMemcpyHostToDevice);//copy cell values to GPU
	printf("Memcpy dX: %s\n", cudaGetErrorString(errCode));
	return d_nodes;
	//~~ALB
}

Mat2D* sendActual(Mat2D* actual) {
	/*A function to send the training set output values to the GPU for comparison
	
	*/
	Mat2D* d_act = (Mat2D*)malloc(sizeof(Mat2D));
	printf("2: %f, 4: %f\n", actual->cells[2], actual->cells[4]);
	cudaError_t errCode = cudaMalloc(&d_act->cells, actual->rows *  actual->columns * sizeof(float));
	printf("GPU cudaMalloc Actual Values: %s\n", cudaGetErrorString(errCode));
	d_act->rows = actual->rows;
	d_act->columns = actual->columns;
	errCode = cudaMemcpy(d_act->cells, actual->cells, actual->rows * actual->columns * sizeof(float), cudaMemcpyHostToDevice);
	printf("Memcpy Actual Nodes: %s\n", cudaGetErrorString(errCode));
	return d_act;
	//~~ALB
}

Mat2D* nodeRetrieve(Mat2D* d_nodes, Mat2D* nodes, bool free = true) {
	/*Code for retrieving layer arrays from GPU
	*/
	Mat2D* first = nodes;// this will be the output

	//Get values from GPU
	cudaError_t errCode = cudaMemcpy(first->cells, d_nodes->cells, d_nodes->rows * d_nodes->columns * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Retrieving nodes from GPU: %s\n", cudaGetErrorString(errCode));
	errCode = cudaMemcpy(first->dTh, d_nodes->dTh, d_nodes->rows * d_nodes->columns * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Retrieving updates from GPU: %s\n", cudaGetErrorString(errCode));

	//Debug code (see very bottom of this file for more helpful debug code)
	errCode = cudaGetLastError();
	if (errCode != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(errCode));
		exit(-1);
	}


	Mat2D* d_temp = d_nodes; // for freeing if applicable
	d_nodes = d_nodes->next;//move through linked list

	if (free) {
		cudaFree(d_temp->cells);//free gpu memory if applicable
		cudaFree(d_temp->dTh);
		cudaFree(d_temp->dX);
	}
	nodes = nodes->next;//move through nodes

	while (nodes != NULL) {
		//get values from GPU... everything in this loop is basically the same as above
		errCode = cudaMemcpy(nodes->cells, d_nodes->cells, d_nodes->rows * d_nodes->columns * sizeof(float), cudaMemcpyDeviceToHost);
		printf("Retrieving nodes from GPU: %s\n", cudaGetErrorString(errCode));
		errCode = cudaMemcpy(nodes->dTh, d_nodes->dTh, d_nodes->rows * d_nodes->columns * sizeof(float), cudaMemcpyDeviceToHost);
		printf("Retrieving updates from GPU: %s\n", cudaGetErrorString(errCode));
		d_temp = d_nodes;
		d_nodes = d_nodes->next;
		if (free) {
			cudaFree(d_temp->cells);//free gpu memory if applicable
			cudaFree(d_temp->dTh);
			cudaFree(d_temp->dX);
		}
		nodes = nodes->next;
	}
	return first;
	//~~ALB
}

Mat2D* pNodesSetup(Mat2D* nodes) {
	/*Code for processing the nodes.
	  This should eventually be improved with aSync.
	  The idea here is to be able to try to keep the transfers to the GPU to a minimum
	  Eventually some of this code should be put onto the GPU
	  */
	printf("\n\n============= Process Nodes Setup ===============\n");
	Mat2D* first = nodes;
	Mat2D* next = first->next;

	Mat2D* d_first = lmatSend2D(first); //send first, and get back pointer to device-first
	Mat2D* d_next = lmatSend2D(next); //send next, and get back pointer to device-next
	d_first->next = d_next; // yep, we're creating a linked list to retrieve everything later

	dim3 tPb(BLKSZ, BLKSZ); //standard tpb code
	dim3 nb((int)ceil((double)d_next->columns / tPb.x), (int)ceil((double)d_next->rows / tPb.y));
	//dim3 nb((d_next->rows + tPb.y) / tPb.y, (d_next->columns + tPb.x) / tPb.x);

	printf("Layer 1\n"); //I didn't print a layer 0 because nothing is going on there
	logistic2D <<<nb, tPb >>> (*d_next, *d_first); // kernal execution
	cudaError_t errCode = cudaThreadSynchronize(); //sync threads
	printf("GPU Thread Synchronization: %s\n", cudaGetErrorString(errCode));


	Mat2D* d_prev = d_next;
	next = next->next;//move through linked list

	int i = 2;//for tracking in the print code
	while (next != NULL) { //go until end of host linked list
		d_next = lmatSend2D(next);//send next layer
		d_prev->next = d_next;//building device linked list

		//basically repeated from above
		dim3 nb(ceil((double)d_next->columns / tPb.x), ceil((double)d_next->rows / tPb.y));
		//dim3 nb((d_next->rows + tPb.y) / tPb.y, (d_next->columns + tPb.x) / tPb.x);
		printf("Layer %i\n", i);
		logistic2D <<<nb, tPb >>> (*d_next, *d_prev);
		errCode = cudaThreadSynchronize();
		printf("GPU Thread Synchronization: %s\n", cudaGetErrorString(errCode));

		//next
		d_prev = d_next;
		next = next->next; //move through linked list
		i++;
	}
	d_prev->next = NULL;
	d_first->end = d_prev;
	return d_first; //return device linked list so we can get the results later
					//~~ALB
}

Mat2D* uNodes(Mat2D* d_nodes, float alpha) {
	/*Code to update the node weights during the learning cycle
	*/

	Mat2D* t = d_nodes;
	printf("==================STARTING NODE UPDATE========================= \n");
	int i = 0;
	//go through linked list of layer arrays
	while (t != NULL) {
		dim3 tPb(BLKSZ, BLKSZ);
		dim3 nb(ceil((double)t->columns / tPb.x), ceil((double)t->rows / tPb.y));
		//dim3 nb((t->rows + tPb.y) / tPb.y, (t->columns + tPb.x) / tPb.x);
		
		printf("\nLayer %i Update \n", i);
		updateNodes<<<nb, tPb>>>(*t, alpha);
		
		cudaError_t errCode = cudaDeviceSynchronize();
		printf("\nNode Update: %s\n", cudaGetErrorString(errCode));
		
		errCode = cudaGetLastError();
		if (errCode != cudaSuccess)
		{
			fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(errCode));
			exit(-1);
		}
		t = t->next;
		i++;
	}
	cudaError_t errCode = cudaDeviceSynchronize();
	printf("\nNode Update: %s\n", cudaGetErrorString(errCode));
	printf("==================NODE UPDATE COMPLETE========================= \n");
	return d_nodes;
	//~~ALB
}

Mat2D* changeIn(Mat2D* d_nodes, Mat2D* inputs, int r) {
	//function to change the input layer for each cycle through
	printf("-Change Inputs-\n");
	Mat2D* d_in = (Mat2D*)malloc(sizeof(Mat2D));
	d_in->cells = (float*)malloc(sizeof(inputs->cells));

	int i = 0;
	for (; i < inputs->columns;) {
		d_in->cells[i] = inputs->cells[r * inputs->columns + i];
		r++;
		i++;
	}
	cudaFree(d_nodes->cells);
	cudaError_t errCode = cudaMalloc(&d_nodes->cells, sizeof(inputs));
	printf("Input change malloc: %s\n", cudaGetErrorString(errCode));
	errCode = cudaMemcpy(d_nodes->cells, d_in->cells, sizeof(inputs), cudaMemcpyHostToDevice);
	printf("Input change memcpy: %s\n\n", cudaGetErrorString(errCode));
	return d_nodes;
	//~~ALB
}

Mat2D* pNodes(Mat2D* d_n, bool learn = false, Mat2D* actual = NULL, int run = 0, Mat2D* last = NULL) {
	/* Main processing function... 

	Travels through the linked list, calling the main logistic forward function
	Reaches the end and pivots, comparing the output with the actual value in the training set
	Travels backwards through the recursion return path and calculates the change to the node weights
	
	*/

	//Travels through the linked list, calling the main logistic forward function
	if (d_n->next != NULL) {
		printf("-Calc Forward-\n");
		dim3 tPb(BLKSZ, BLKSZ);
		dim3 dimGrid(ceil((double)d_n->columns / tPb.x), ceil((double)d_n->rows / tPb.y));
		//dim3 nb((d_n->rows + tPb.y) / tPb.y, (d_n->columns + tPb.x) / tPb.x);
		logistic2D <<< dimGrid, tPb >>> (*d_n->next, *d_n);
		cudaError_t errCode = cudaDeviceSynchronize();
		//printf("Calc Forward STATUS: %s\n\n", cudaGetErrorString(errCode));
		d_n = pNodes(d_n->next, learn, actual, run, d_n);
	}

	//Reaches the end and pivots, comparing the output with the actual value in the training set
	else if (learn) {
		//printf("-Calc Pivot-\n");
		dim3 tPb(BLKSZ, BLKSZ);
		dim3 dimGrid(ceil((double)d_n->columns / tPb.x), ceil((double)d_n->rows / tPb.y));
		//dim3 nb((d_n->rows + tPb.y) / tPb.y, (d_n->columns + tPb.x) / tPb.x);
		outPivotLog <<< dimGrid, tPb >>> (*last, *d_n, *actual, run);
		cudaError_t errCode = cudaDeviceSynchronize();
		//printf("Calc Pivot STATUS: %s\n\n", cudaGetErrorString(errCode));
	}
	

	//Travels backwards through the recursion return path and calculates the change to the node weights
	if (learn && d_n->next != NULL && last != NULL) {
		//printf("-Calc Backwards-\n");
		//printf("dX:\n");
		dim3 tPb(BLKSZ, BLKSZ);
		dim3 dimGrid(ceil((double)d_n->columns / tPb.x), ceil((double)d_n->rows / tPb.y));
		//dim3 nb((d_n->rows + tPb.y) / tPb.y, (d_n->columns + tPb.x) / tPb.x);
		nodeBackwardLog <<< dimGrid, tPb >>> (*last, *d_n, *d_n->next);
		cudaError_t errCode = cudaDeviceSynchronize();
		//printf("\n Calc Backwards STATUS: %s\n\n", cudaGetErrorString(errCode));
	}

	if (last != NULL) {
		return last;
	}
	else {
		return d_n;
	}
	//~~ALB
}

float pullBatchErr(Mat2D* d_last, float bSize) {
	//float* out = (float*)malloc(d_last->columns * sizeof(float));
	Mat2D last;
	last.dX = (float*)malloc(d_last->columns * d_last->rows * sizeof(float));
	cudaError_t errCode = cudaMemcpy(last.dX, d_last->dX, d_last->columns * d_last->rows * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Pulling output from last node: %s\n", cudaGetErrorString(errCode));

	
	/*for (int i = 0; i < d_last->columns; ++i) {
		out[i] = last.cells[i];
	}*/
	float out = last.dX[d_last->columns]/ bSize;
	free(last.dX);
	return out;
}

int main(int argc, char* argv) {
	//These testers for the matrix math will eventually be gone
	//float a[] = {
	//	2, 4, 6, 7,
	//	1, 3, 4, 6, 9
	//};
	//int rA = 3;
	//int cA = 3;
	//float b[] = {
	//	0, 4,
	//    2, 3, 
	//	2, 4,
	//	4, 4, 3
	//};
	//int rB = 3;
	//int cB = 3;

	printf("============= Initializing ===============\n");
	cudaDeviceProp Dev;
	if (cudaGetDeviceProperties(&Dev, 0) == cudaSuccess) {
		printf("................Hardware Properties................\n");
		printf(".....Device Name: %s\n", Dev.name);
		printf(".....Compute Version: %i\n", Dev.major);
		printf(".....Max Clock Rate (MHz): %f\n", (float)Dev.clockRate / (float)(1000 * 1000));
		printf(".....Total Global Memory (MB): %f\n", (float)Dev.totalGlobalMem / (float)(1000*1000));
		printf(".....Total Shared Memory Per Block (kB): %f\n", (float)Dev.sharedMemPerBlock / (float)(1000));
		printf(".....Max Threads Per Block: %i\n", Dev.maxThreadsPerBlock);
		printf(".....Max Thread Dim: %i\n", *Dev.maxThreadsDim);
		printf(".....Max Grid size: %i\n", *Dev.maxGridSize);
		printf(".....# of MultiProcessors: %i\n", Dev.multiProcessorCount);
		printf(".....Max Threads Per MultiProcessor: %i\n", Dev.maxThreadsPerMultiProcessor);
		printf("...................................................\n");
	}
	Config setup("");
	printf("in: %s, act: %s, out: %s\n", setup.in.c_str(), setup.act.c_str(), setup.out.c_str());
	Timer Time(setup.timer);

	//Below is a layer setup variable, this takes an array where each value is the number of nodes in that layer
	laySet lay;
	// int x[] = { 1, 24,  1 };
	/*lay.layers = sizeof(x) / sizeof(int);
	lay.nPl = x;*/
	lay.layers = setup.layers;
	lay.nPl = setup.nodesPerlayer;
	

	//below is the training set inputs as a vector (turned into a Mat2D)... this is currently just hard-coded.
	//each row is the next input for the traning set
	Mat2D* inputs = csvToMat2D(setup.in, lay.nPl[0]);

	//below is the training actual values set as a vector... this is currently just hard-coded.
	//each row is the next actual value for the traning set
	Mat2D* actual = csvToMat2D(setup.act, lay.nPl[lay.layers - 1]);
	//actual->columns = lay.nPl[lay.layers -1];
	//float act[] = { 0.5, 2.5, 1.5, 1, 1.5, 2, 2.5, 0.5, 0.5, 2.5, 1.5, 1, 1.5, 2, 2.5, 0.5, 0.5, 2.5, 1.5, 1, 1.5, 2, 2.5, 0.5 };
	//actual->rows = sizeof(act) / sizeof(float);
	//vecToMat2DP(act, actual);
	
	int bSize = setup.batchSize; //batch size.... the number of runs before updating the variables
	float alpha = setup.alpha;
	string oString = setup.out; //output file name

	Mat2D* d_in = (Mat2D*)malloc(sizeof(Mat2D));
	d_in->cells = (float*)malloc(inputs->columns * inputs->rows * sizeof(float));
	d_in->columns = inputs->columns;
	d_in->rows = inputs->rows;

	cudaError_t errCode = cudaMalloc(&d_in->cells, inputs->columns * inputs->rows * sizeof(float));
	printf("Allocate for inputs: %s\n\n", cudaGetErrorString(errCode));
	Time.lap("Allocate inputs");
	errCode = cudaMemcpy(d_in->cells, inputs->cells, inputs->columns * inputs->rows * sizeof(float), cudaMemcpyHostToDevice);
	printf("Send inputs: %s\n\n", cudaGetErrorString(errCode));
	Time.lap("Send Inputs");
	/*inputs->columns = x[0];
	float inp[] = { 1, 5, 3, 2, 3, 4, 5, 1, 1, 5, 3, 2, 3, 4, 5, 1, 1, 5, 3, 2, 3, 4, 5, 1 };
	inputs->cells = (float*)malloc(sizeof(inp));
	inputs->cells = inp;*/


	Mat2D* first = hiddenSetup(lay); //setup the hidden layers based on the layset variable lay
	Mat2D* d_first = pNodesSetup(first); //initial run through with all inputs set to one
	Mat2D* d_act = sendActual(actual); //send the training set outputs to the GPU

	ofstream outFile;
	outFile.open(oString);
	outFile << "Batch #, ";
	for (int i = 0; i < d_act->columns; ++i) {
		outFile << "Output Node " << i << " error squared, ";
	}
	

	//now we run through all of the training set... This will later be replaced by other options as to when 
	//the learning stops
	float bErr;
	
	printf("\n\n========================================================== Begin Batch Run =====================================================\n");
	printf("---Batch Size: %i\n", bSize);
	for (int cyc = 0; cyc < 1; ++cyc) {
		int rn = 0;
		for (; rn < inputs->columns * inputs->rows;) {
			int b = bSize;
			while (b > 0) { //this loops until the batch size is met and then moves to the update step
				printf("\n--Run: %i\n", rn + b*cyc);
				//d_first = changeIn(d_first, inputs, rn); //change the inputs to the corresponding training set
				dim3 tPb(BLKSZ, BLKSZ);
				dim3 nb(ceil((double)d_first->rows/ tPb.y), ceil((double)d_first->columns/tPb.x));
				changeInK <<< nb, tPb >>> (*d_first, *d_in, rn);
				errCode = cudaDeviceSynchronize();
				printf("Calc Update Node: %s\n\n", cudaGetErrorString(errCode));
				d_first = pNodes(d_first, true, d_act, rn); //process the nodes
				b = b - 1; //next batch index
				rn = rn + 1; //next run index
			}
			bErr = pullBatchErr(d_first->end, (float)bSize);
			//outToCsv(nOut, actual, outFile, rn);
			//
			//float errSq;
			int c = actual->columns;
			outFile << endl << (rn / inputs->columns)/bSize;
			for (int i = 0; i < c; ++i) {
				//errSq = (nOut[i] - actual->cells[rn*c + i]);
				//errSq *= errSq;
				outFile << ", " << bErr;
			}
			//
			d_first = uNodes(d_first, alpha); //update the node weights
		}

	}

	printf("\n============================================================= End Batch Run ======================================================\n");
	errCode = cudaGetLastError();
	printf("Starting nodeRetrieve, last CUDA error: %s\n", cudaGetErrorString(errCode));

	first = nodeRetrieve(d_first, first);
	
	Mat2D* temp;
	int i = 0;
	while (first != NULL) {
		printf("Layer %i\n", i);
		print2DMat(*first);
		temp = (Mat2D*)first;
		first = first->next;
		free(temp);
		i++;
		
	}

	cudaFree(d_in);
	free(d_in);
	//outFile->close;
//++++++++!!!!!!!! If having errors on external code, the reset below may cause it================
	errCode = cudaDeviceReset(); //clear any remaining items on device...
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



~~ALB
*/