import numpy as np
import pandas as pd

# K = -0.6
# alpha = -0.25

# results = []

# for Kt in np.linspace(-100, 100, 20001):
#     coeffs = [1, 1 + Kt, 1, 1 + Kt + K, K * alpha]
#     poles = np.roots(coeffs)

#     real_parts = np.sort(np.real(poles))
#     results.append([Kt, *real_parts])

# df = pd.DataFrame(
#     results,
#     columns=["Kt", "Pole1", "Pole2", "Pole3", "Pole4"]
# )

# df.to_csv("pole_sweep.csv", index=False)


A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
])

det = np.linalg.det(A)

print("Determinant:", det)
print("Invertible:", not np.isclose(det, 0))

import numpy as np

A = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0]
])

det = np.linalg.det(A)

print("Determinant:", det)
print("Invertible:", not np.isclose(det, 0))