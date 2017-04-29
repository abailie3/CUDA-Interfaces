
//#include <Python.h>
//#include <ctype.h>
//
//typedef struct {
//	int rows, cols;
//	float* data;
//} StructureArray;
//
//float fromPy(int argc, char *argv[]) {
//	printf("asdf");
//	PyObject *pName, *pModule, *pDict, *pFunc;
//	PyObject *pArgs, *pValue;
//	Py_Initialize();
//	if (argv[0] != "") {
//		PyObject *sysPath = PySys_GetObject("path");
//		PyObject *path = PyUnicode_DecodeFSDefault(argv[0]);
//		PyList_Insert(sysPath, 0, path);
//	}
//	pName = PyUnicode_DecodeFSDefault(argv[1]);
//	pModule = PyImport_Import(pName);
//
//	if (pModule != NULL) {
//		pFunc = PyObject_GetAttrString(pModule, argv[2]);
//		if (pFunc && PyCallable_Check(pFunc)) {
//			pArgs = PyTuple_New(argc - 3);
//			for (int i = 0; i < argc - 3; ++i) {
//				pValue = PyLong_FromLong(atoi(argv[i + 3]));
//				if (!pValue) {
//					Py_DECREF(pArgs);
//					Py_DECREF(pModule);
//					fprintf(stderr, "Cannot convert arg\n");
//					return -1;
//				}
//
//				PyTuple_SetItem(pArgs, i, pValue);
//			}
//			pValue = PyObject_CallObject(pFunc, pArgs);
//			
//			//float* tests = (float*)PyObject_CallObject(pFunc, pArgs);
//			//StructureArray* test = (StructureArray*)PyObject_CallObject(pFunc, pArgs);
//			StructureArray* test = (StructureArray*)PyLong_AsLong(pValue);
//			printf("did this work %d\n", test->cols);
//			//printf("%d", tests);
//			Py_DECREF(pArgs);
//			if (pValue != NULL) {
//				printf("Result of call: %ld\n", PyLong_AsLong(pValue));
//				//printf("Result of call: %ld\n", PyList_AsTuple(pValue));			
//				Py_DECREF(pValue);
//			} else {
//				Py_DECREF(pFunc);
//				Py_DECREF(pModule);
//				PyErr_Print();
//				fprintf(stderr, "Call Failed\n");
//				return -1;
//			}
//		} else {
//			if (PyErr_Occurred())
//				PyErr_Print();
//			fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
//		}
//		Py_XDECREF(pFunc);
//		Py_DECREF(pModule);
//	} else {
//		PyErr_Print();
//		fprintf(stderr, "Failed to load\"%s\"\n", argv[1]);
//		return -1;
//	}
//	Py_Finalize();
//	return 0;
//
//}