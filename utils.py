import numpy as np
import numba

@numba.njit
def jit_to_categorical(ndarray, classes: int) -> np.ndarray:
    return np.array([[0 if val!=i else 1 for i in range(classes)] for val in ndarray])

@numba.njit
def jit_from_categorical(ndarray: np.ndarray) -> np.ndarray:
    outputs = np.empty(0)
    for arr in ndarray:
        outputs = np.append(outputs, (arr == max(arr)).nonzero())
    return outputs.astype(int)

@numba.njit
def jit_categorical_compare(output_layer, expected):
    return output_layer.argsort()[-1]==expected.argsort()[-1]

class OutSplit(object):
    def __init__(self, filename):
        self.file = open("".join(['data/', filename]), 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
    
    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        # self.file.flush()
        self.stdout.flush()

@numba.njit
def jit_round_compare(output_layer, expected):
    truth = 1
    for i, out in enumerate(output_layer):
        truth += not round(out) == expected[i]
    return truth == 1