import numpy as np

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=(b != 0) | (a == 0))

# Example arrays
a = np.array([0, 2, 3, 4])
b = np.array([1, 0, 0, 4])

result = safe_divide(a, b)
print(result)