cdef extern void compute_icdf(
        float(*callback)(float, void*),
        float a, float b, unsigned n, float eps,
        float *result, void *data)
