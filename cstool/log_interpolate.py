import numpy as np


def log_interpolate_1d(x_i, y_i):
    """Creates a function that interpolates on the function values given
    in the input arrays `x_i` and `y_i`. The interpolation is logarithmic
    in the range."""
    assert len(x_i.shape) == 1
    assert x_i.shape == y_i.shape

    x_steps = np.log(x_i[1:] / x_i[:-1])

    def interpolated(x):
        idx = np.searchsorted(x_i, x) - 1
        w = np.log(x / x_i[idx]) / x_steps[idx]
        y = (1 - w) * y_i[idx] + w * y_i[idx + 1]
        return y

    return interpolated


if __name__ == "__main__":
    x_i = np.logspace(-2, 2, 10)
    y_i = np.log(x_i)**2
    f = log_interpolate_1d(x_i, y_i)

    # output suitable for GnuPlot:
    # > set log x
    # > plot '< python3 log_interpolate.py' i 0 w p, '' i 1 w l

    # print original points
    for i in np.c_[x_i, y_i]:
        print(i[0], i[1])

    print("\n")

    x = np.logspace(-2, 2, 100)
    y = f(x)
    # print interpolated points
    for i in np.c_[x, y]:
        print(i[0], i[1])
