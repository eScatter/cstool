#include <algorithm>
#include "tabulated.hh"

using namespace eScatter;
using namespace numeric;

extern "C" void compute_icdf(
        float (*pdf)(float),
        float a,
        float b,
        unsigned n,
        float epsilon,
        float *data)
{
    auto table = Tabulated<float>::inverse_cdf(pdf, a, b, n, epsilon);
    std::copy(table.values.begin(), table.values.end(), data);
}
