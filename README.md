# Machine Learning/Neural Nets
---Where the machines aren't the only ones learning--

Hello!

  This is my journey to learn machine learning/AI/neural networks and to learn how to code. I'm trying to teach myself how to code in C/C++ and eventually python/Java/others. The only formal training I have is a Matlab for Engineers class
in college, SO if you see something that I'm doing is a "no-no" or just plain dumb, please let me know!!

  The idea for this project is to build a machine learning/neural network toolkit from the ground up (well from what I consider the ground...). It will perform calculations on the GPU, utilizing Nvidia's CUDA toolkit. (For reference, I currently have a GTX 560) Yes, I know there are a wealth of already made neural network tools (tensorflow, etc.), but this is primarily a learning excercise to try to really understand how this all works.

  I'll try to comment my code the best I can, so it might be some use to someone else.

-Austin


============ Change Log ===================

v0: 1/15/2017 
* original

v0.01: 1/15/2017	
* added transpose

v0.1: 1/21/2016		
* added various matrix math functions
* added neural network architecture:
	* added logistic2D kernel
	* added lmatSend2D
	* added nodeRetrieve
	* added processNodes
	* added layersetup
	* added hiddenSetup
* current neural network support runs with 0 errors on Cuda-memcheck

v0.2: 1/29/2017		
 * implemented working neural network functionality:
	* added nodeBackwardLog kernel
	* added outPivotLog kernel
	* added updateNodes kernel
	* added sendActual support function
	* addded uNodes support function
	* added changeIn support function
	* added pNodes process function
	* tweaked most of the previously added neural network architecture
	* changed main function to run the neural network
* current technology will successfuly perform batch gradient descent with 0 errors on Cuda-memcheck   

===========================================
