import numpy as np
from typing import NoReturn

def check_same_number_of_rows(
        array1: np.ndarray, 
        array2: np.ndarray
        ) -> NoReturn:
    """
    Checks if the number of rows in array1 and array2 are the same.

    Raises a ValueError the number of rows are different. 
    """
    if array1.shape[0] != array2.shape[0]:
        raise ValueError(
            f"Arrays have different number of rows: {array1.shape[0]} != {array2.shape[0]}"
        )

def softmax(
        x: np.ndarray
        ) -> np.ndarray:
    """
    Retruns the sofmax function of the array x.

    Keyword arguments:
        x (np.ndarray):
            The argument of the softmax function.

    Returns (np.ndarray):
        The value of the softmax function.
    """

    exp_x = np.exp(x)
    Z_inv = (1./np.sum(exp_x, axis = 1)).reshape(-1,1)

    return Z_inv*exp_x

def one_hot_encoder(
        a: np.ndarray, 
        class_labels: np.ndarray
        ) -> np.ndarray:
    """
    Retruns the one-hot encode of the array a.

    Keyword arguments:
        a (np.ndarray): array of shape (n_samples,)
            The array to be encoded.

    Returns (np.ndarray): array of shape (n_samples, n_classes)
        The on-hot enconded array a.
    """

    n_classes = class_labels.shape[0]
    n_samples = a.shape[0]
    
    a_hot_encoded = np.zeros((n_samples, n_classes), int)
    for i in range(n_classes):
        
        idx = np.where(a == class_labels[i])[0].astype(int)
        a_hot_encoded[idx, i] = 1

    return a_hot_encoded

def kronecker_delta(
        i: int, 
        j: int
        ) -> int:
    """
    Retruns the Kronecker delta with arguments i, j.

    Keyword arguments:
        i,j (int):
            The arguments of the Kronecker delta.

    Returns (int):
        0 if the arguments are different and 1 otherwise.
    """
    
    if i == j:
        return 1
    else:
        return 0
