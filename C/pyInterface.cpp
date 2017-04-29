//#ifndef __PYINTERFACE_CPP_INCLUDED__
//#define __PYINTERFACE_CPP_INCLUDED__
//#include "neuralnetKernal.h"
//#include <Python.h>
//
//PyObject* nnTest(PyObject *) {
//  return PyFloat_FromDouble((double)neural_main());
//  //return PyFloat_FromDouble(42.);
//}
//
//static PyMethodDef neuralnet_methods[] = {
//  {"nnTest_C", (PyCFunction)nnTest, METH_O, nullptr},
//  {nullptr, nullptr, 0, nullptr}
//};
//
//static PyModuleDef neuralnet_module = {
//  PyModuleDef_HEAD_INIT,
//  "neuralnet", // Name
//  "C++ neural net functions utilizing CUDA framework", // Description
//  0,
//  neuralnet_methods // the methods
//};
//
//PyMODINIT_FUNC PyInit_neuralnet() {
//  return PyModule_Create(&neuralnet_module);
//}
//#endif













//#pragma once
//#include "pyInterface.h"
//
//int main(int argc, char *argv[]) {
//  wchar_t *program = Py_DecodeLocale(argv[0], NULL);
//  if (program == NULL) {
//    if (neural_main() == 0) {
//      fprintf(stderr, "Running From neuralMain");
//      exit(EXIT_SUCCESS);
//    } else {
//      fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
//      exit(EXIT_FAILURE);
//    }
//
//  }
//
//  PyImport_AppendInittab("neuralnet", PyInit_neuralnet);
//  Py_SetProgramName(program);
//  Py_Initialize();
//  PyImport_ImportModule("neuralnet");
//
//  //CODE HERE
//  neural_main();
//  PyMem_RawFree(program);
//  return 0;
//
//}