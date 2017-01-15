/*
========= CUDA matrix typedefs v0 =========
			By: Austin Bailie

Matrix typedefs with support for higher
dimension matricies.
test
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
#include <stdio.h>

typedef struct { //row then column
	int rows;
	int columns;
	float* cells;
} Mat2D;

typedef struct { //row then column then level
	int rows;
	int columns;
	int levels;
	float* cells;
} Mat3D;

typedef struct { //row then column then level then time
	int rows;
	int columns;
	int levels;
	int time;
	float* cells;
} Mat4D;

typedef struct { //row then column then level then time then fractalPlane
	int rows;
	int columns;
	int levels;
	int time;
	int fractalPlane;
	float* cells;
} Mat5D;