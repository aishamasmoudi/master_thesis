import numpy as np

def string_to_bool_array(s):
    # Remove brackets and split on whitespace
    bool_strings = s.strip('[]').split()
    # Convert each string to actual boolean
    return np.array([b == 'True' for b in bool_strings])


def compute_meanHR(group):
    array = np.stack(group.to_numpy())
    return np.mean(array, axis=0)

