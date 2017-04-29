#ifndef __JAVAINPUTCONVERSION_H_INCLUDED__
#define __JAVAINPUTCONVERSION_H_INCLUDED__

#include "nodeSet.h"

LaySet JArrayToLayset(int* in_arr, int layers) {
	LaySet out;
	out.nPl = in_arr;
	printf("in_arr \n");
	out.layers = layers;
	return out;
}

Mat2D* JArrayToInMat(float* in_arr, int rows, int columns) {

	Mat2D* out = newMat2D(rows, columns);
	out->cells = in_arr;
	return out;
	//values.size() / columns, columns
}

#endif