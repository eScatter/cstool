#include <algorithm>
#include <functional>
#include "tabulated.hh"

using namespace eScatter;
using namespace numeric;
using namespace std::placeholders;

extern "C" void compute_icdf(
        float (*pdf)(float, void*),
        float a,
        float b,
        unsigned n,
        float epsilon,
        float *result,
        void *data)
{
    auto table = Tabulated<float>::inverse_cdf(
        std::bind(pdf, _1, data), a, b, n, epsilon);
    std::copy(table.values.begin(), table.values.end(), result);
}
