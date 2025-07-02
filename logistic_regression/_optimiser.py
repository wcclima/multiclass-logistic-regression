import numpy as np
from functools import partial
from ._utilities import softmax, kronecker_delta


def gradient_func(
        X: np.ndarray, 
        y: np.ndarray, 
        sample_weight: np.ndarray, 
        b_hat: np.ndarray
        ) -> np.ndarray:
    """
    Returns the value of the gradient of the log-likelyhood function 
    in the logistic regression problem with respect to the 
    regression coefficients.

    Keyword arguments:
        X (np.ndarray): array of shape (n_samples, n_predictors).
            The predictor's data.

        y (np.ndarray): array of shape (n_samples,).
            The dependent variable's data.

        sample_weight (np.ndarray): array of shape (n_samples,) 
            The weight of each sample.

        b_hat (np.ndarray): array of shape (n_predictors, n_classes)
            The regression coefficients

    Returns (np.ndarray): array of shape (n_predictors*(n_classes - 1), 1)
        The the gradient of the log-lilkelyhood function 
        evaluated at b_hat.    
    """

    k = b_hat.shape[1]
    W = np.diag(sample_weight)

    grad = np.reshape(X.T @ W @ (y - softmax(X @ b_hat))[:, :k - 1], newshape = (-1, 1), order = 'F')

    return grad


def hessian_func(
        X: np.ndarray, 
        sample_weight: np.ndarray, 
        b_hat: np.ndarray
        ) -> np.ndarray:    
    """
    Returns the value of the Hessian matrix of the log-likelyhood 
    function in the logistic regression problem with respect to 
    the regression coefficients.

    Keyword arguments:
        X (np.ndarray): array of shape (n_samples, n_predictors).
            The predictor's data.

        y (np.ndarray): array of shape (n_samples,).
            The dependent variable's data.

        sample_weight (np.ndarray): array of shape (n_samples,) 
            The weight of each sample.

        b_hat (np.ndarray): array of shape (n_predictors, 1)
            The regression coefficients

    Returns (np.ndarray): array of shape (n_predictors*(n_classes - 1), n_predictors*(n_classes - 1))
        The the Hessian matrix of the log-lilkelyhood function 
        evaluated at b_hat.
    """

    k = b_hat.shape[1]
    H = []
    for i in range(k - 1):
        h = []
        for j in range(k - 1):
            S_ij = np.diag(sample_weight*softmax((X @ b_hat))[:, i]*(kronecker_delta(i, j) - softmax(X @ b_hat)[:, j]))
            h.append(X.T @ S_ij @ X)

        H.append(h)


    return np.block(H)


class NewtonOptimiser(object):
    """
    Newton-Raphson optimiser.

    NewtonOptimiser finds the extremum of the log-likelyhood 
    function for a logistic regression model by searching for 
    the zero of the norm of the function gradient.

    
    Parameters:
        X (np.ndarray): array of shape (n_samples, n_predictors).
            The predictor's data.

        y (np.ndarray): array of shape (n_samples,).
            The dependent variable's data.

        sample_weight (np.ndarray, default = None): array of shape (n_samples,) 
            The weight of each sample.

    Attributes:
        vec_func (callable):
            A function that takes a 1D np.ndarray (shape (n,)) input and 
            returns a 1D np.ndarray (shape (n,)) output. Represents the 
            vector-valued function whose root (zero) is sought.
            Example: vec_func(z) -> np.array([f1(z), f2(z), ..., fn(z)])

        inv_jacobian_func (callable):
            A function that takes a 1D np.ndarray (shape (n,)) input and 
            returns a 2D np.ndarray (shape (n, n)) output. Represents the 
            inverse matrix of the Jacobian (∇F) of F evaluated at a point 
            z.

    Methods:
        optimise(z0, precision, max_iter):
            Performs the optimisation.
    """

    def __init__(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
            sample_weight: np.ndarray
            ) -> None:

        self.vec_func = partial(gradient_func, X, y, sample_weight)
        self.jacobian_func = partial(hessian_func, X, sample_weight)

        self.n_iter_ = 0


    def optimise(
            self, 
            z0: np.ndarray, 
            tolerance: float, 
            max_iter: int
            ) -> np.ndarray:
        """
        Newton-Raphson method to find the zero of the norm of the vector 
        function vec_func, up to a certain precision.

        Keyword arguments:
            z0: array_like, shape (n,)
                    Initial guess for the root.

            tolerance: (float)
                The tolerance for the stopping criterion. Iteration stops when the 
                norm of vec_func is less than `tolerance`.

            max_iter: (int)
                Maximum number of iterations to perform.


        Returns:
            z : ndarray, shape (n,)
                Estimated root of vec_func, such that norm of vec_funct ≈ 0.

        Raises:
            ValueError
                If the method does not converge within `max_iter` iterations.

        Notes:
            This method assumes that the inverse Jacobian is provided and is accurate.
            It does not compute derivatives numerically.
        """

        p = z0.shape[0]

        z = z0
        f = self.vec_func(z)
        f_norm = np.sqrt((f.flatten()**2).sum())

        while f_norm > tolerance:

            J_inv = np.linalg.inv(self.jacobian_func(z))
            delta_z = J_inv @ f
            delta_z = np.append(delta_z, np.zeros((p, 1)), axis = 0)

            z += np.reshape(delta_z, z0.shape, 'F')

            f = self.vec_func(z)
            f_norm = np.sqrt((f.flatten()**2).sum())

            self.n_iter_ += 1
            if self.n_iter_ > max_iter:
                raise ValueError(f"Newton-Raphson did not converge within {max_iter} iterations.")
            

        return z
