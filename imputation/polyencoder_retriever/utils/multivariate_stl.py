from statsmodels.tsa.seasonal import STL
import numpy as np


def Multivariate_STL(data, seasonal=13, period=24):
    T, C = data.shape
    stl_results = [
        STL(data[:, c], seasonal=seasonal, period=period).fit() for c in range(C)
    ]
    reconstructed = [res.trend + res.seasonal for res in stl_results]

    return np.stack(reconstructed, axis=1)
