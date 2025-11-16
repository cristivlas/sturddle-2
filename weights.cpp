#include <Python.h>
#include "common.h"

extern "C" {
    #include "weights.h"
}

// Export as PyCapsules for cross-platform access
static PyObject* get_spatial_attn_w(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)spatial_attn_w, "spatial_attn_w", nullptr);
}

static PyObject* get_spatial_attn_b(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)spatial_attn_b, "spatial_attn_b", nullptr);
}

static PyObject* get_hidden_1a_w(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)hidden_1a_w, "hidden_1a_w", nullptr);
}

static PyObject* get_hidden_1a_b(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)hidden_1a_b, "hidden_1a_b", nullptr);
}

static PyObject* get_hidden_1b_w(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)hidden_1b_w, "hidden_1b_w", nullptr);
}

static PyObject* get_hidden_1b_b(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)hidden_1b_b, "hidden_1b_b", nullptr);
}

static PyObject* get_hidden_2_w(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)hidden_2_w, "hidden_2_w", nullptr);
}

static PyObject* get_hidden_2_b(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)hidden_2_b, "hidden_2_b", nullptr);
}

static PyObject* get_hidden_3_w(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)hidden_3_w, "hidden_3_w", nullptr);
}

static PyObject* get_hidden_3_b(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)hidden_3_b, "hidden_3_b", nullptr);
}

static PyObject* get_eval_w(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)out_w, "out_w", nullptr);
}

static PyObject* get_eval_b(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)out_b, "out_b", nullptr);
}

#if USE_MOVE_PREDICTION
static PyObject* get_move_w(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)move_w, "move_w", nullptr);
}

static PyObject* get_move_b(PyObject* self, PyObject* args)
{
    return PyCapsule_New((void*)move_b, "move_b", nullptr);
}
#endif /* USE_MOVE_PREDICTION */

static PyMethodDef weights_methods[] = {
    {"get_spatial_attn_w", get_spatial_attn_w, METH_NOARGS, "Get dynamic weights"},
    {"get_spatial_attn_b", get_spatial_attn_b, METH_NOARGS, "Get dynamic biases"},
    {"get_hidden_1a_w", get_hidden_1a_w, METH_NOARGS, "Get hidden_1a weights"},
    {"get_hidden_1a_b", get_hidden_1a_b, METH_NOARGS, "Get hidden_1a biases"},
    {"get_hidden_1b_w", get_hidden_1b_w, METH_NOARGS, "Get hidden_1b weights"},
    {"get_hidden_1b_b", get_hidden_1b_b, METH_NOARGS, "Get hidden_1b biases"},
    {"get_hidden_2_w", get_hidden_2_w, METH_NOARGS, "Get hidden_2 weights"},
    {"get_hidden_2_b", get_hidden_2_b, METH_NOARGS, "Get hidden_2 biases"},
    {"get_hidden_3_w", get_hidden_3_w, METH_NOARGS, "Get hidden_3 weights"},
    {"get_hidden_3_b", get_hidden_3_b, METH_NOARGS, "Get hidden_3 biases"},
    {"get_out_w", get_eval_w, METH_NOARGS, "Get eval weights"},
    {"get_out_b", get_eval_b, METH_NOARGS, "Get eval biases"},
#if USE_MOVE_PREDICTION
    {"get_move_w", get_move_w, METH_NOARGS, "Get move weights"},
    {"get_move_b", get_move_b, METH_NOARGS, "Get move biases"},
#endif /* USE_MOVE_PREDICTION */
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef weights_module = {
    PyModuleDef_HEAD_INIT,
    "weights",
    nullptr,
    -1,
    weights_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

PyMODINIT_FUNC PyInit_weights(void)
{
    return PyModule_Create(&weights_module);
}
