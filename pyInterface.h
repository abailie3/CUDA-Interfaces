#pragma once
#include <Python.h>
static PyObject *
neural_net(PyObject *self, PyObject *args) {
  const char *command;
  int sts;

  if (!PyArg_ParseTuple(args, "s", &command))
    return NULL;
  sts = system(command);
  return Py_BuildValue("i", sts);
}