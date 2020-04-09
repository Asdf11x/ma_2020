import numpy as np
import sys

a = [[650.946, 664.68, 554.876, 537.237, 590.253, 780.22, 0, 0, 633.337, 682.244, 605.876, 713.554, 0, 0, 0, 0, 0, 0]]
b = [5, 5, 5, 5, 5, 5, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0]

a.append(b)

# print(a)

files = np.empty(100, dtype=np.float32)
print(sys.getsizeof(files))
files2 = np.empty(100, dtype=np.float)
print(sys.getsizeof(files2))