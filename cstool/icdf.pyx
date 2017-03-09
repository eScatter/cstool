cimport icdf
import numpy as npp
cimport numpy as np

cdef float callback(float x, void *data):
    return (<object>data)(x)

cpdef inverse_cdf(f, float a, float b, unsigned n, float eps=1e-6):
    data = npp.zeros(n+1, dtype='float32')
    cdef float [:] data_view = data
    compute_icdf(callback, a, b, n, eps, &data_view[0], <void *>f)
    return data


