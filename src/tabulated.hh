#pragma once
#include <vector>
#include <cmath>
#include <iostream>

#include "romberg.hh"
#include "roots.hh"

namespace eScatter { namespace numeric {
    /*! \brief Tabulates values on uniform grid for fast look-up and linear
     * interpolation.
     */
    template <typename real_t>
    class Tabulated
    {
        public:
            std::vector<real_t> values;
            real_t a, b, h;

            Tabulated() {}

            template <typename Range>
            Tabulated(Range const &r, real_t a, real_t b, real_t h):
                values(r.begin(), r.end()), a(a), b(b), h(h) {}

            bool in_bounds(real_t x) const;
            real_t operator()(real_t x) const;

            template <typename Fn>
            static Tabulated inverse_cdf(
                Fn pdf, real_t x1, real_t x2, unsigned n, real_t epsilon);
    };

    /*! \brief Check if value is within bounds of the table and thereby
     * if it is safe to call the interpolation function.
     */
    template <typename real_t>
    bool Tabulated<real_t>::in_bounds(real_t x) const
    {
        return a <= x and x <= b;
    }

    /*! \brief Linear interpolation on the values in the table. This
     * does _not_ perform bounds checks! If bounds checks are needed
     * the caller should call in_bounds method himself.
     */
    template <typename real_t>
    real_t Tabulated<real_t>::operator()(real_t x) const
    {
        real_t w = (x - a) / h;
        unsigned i = floor(w);
        real_t f = w - i;
        return f * values[i + 1] + (1 - f) * values[i];
    }

    /*! \brief Tabulates the inverse CDF of the given PDF.
     *
     * To draw random numbers from a distribution, we need an expression for
     * the inverse of the CDF of that distribution.  Often we do not have an
     * analytic expression for the CDF. This factory function creates a
     * Tabulated object, giving a function with a unit domain and a range from
     * x1 to x2.
     *
     * \param pdf The probability density function.
     * \param x1 Left bound.
     * \param x2 Right bound.
     * \param n  Number of steps in table, should be power of 2.
     * \param epsilon Absolute precision.
     *
     * This method uses Romberg integration to compute the CDF and then
     * Brent-Newton method to find the inverse. We optimise for fast look-up
     * and linear interpolation on a uniform grid. To prevent round-off error
     * we first compute the integral over the entire range, and then subdivide.
     *
     * The final number of elements in the table will be n + 1, including both
     * bounding values.
     */
    template <typename real_t>
    template <typename Fn>
    Tabulated<real_t> Tabulated<real_t>::inverse_cdf(
            Fn pdf, real_t x1, real_t x2, unsigned n, real_t epsilon)
    {
        if ((n & (n - 1)) != 0)
            throw "Tabulated::inverse_cdf needs power-of-two size.";

        Tabulated result;
        result.values.resize(n+1);
        result.values[0] = x1;
        result.values[n] = x2;

        real_t total = integrate_romberg(pdf, x1, x2, epsilon, 16);
        unsigned m = n;
        while (m > 1)
        {
            total /= 2.;
            m /= 2;

            for (unsigned j = m; j < n; j += 2*m)
            {
                x1 = result.values[j - m];
                x2 = result.values[j + m];

                real_t x = find_root_brent(
                    [&] (real_t x) {
                        return integrate_romberg(pdf, x1, x, epsilon, 10);
                    }, pdf, x1, x2, total, epsilon);

                result.values[j] = x;
            }
        }

        result.a = 0.0; result.b = 1.0; result.h = 1.0 / n;
        return result;
    }
}}

