from functools import partial
import numpy as np
from ._utilities import check_same_number_of_rows
from ._utilities import softmax, one_hot_encoder
from ._optimiser import NewtonOptimiser, hessian_func
from scipy.special import erf
from prettytable import PrettyTable


__all__ = ["LogisticRegressor"]

class LogisticRegressor(object):
    """
    Multiclass Logistic Regressor.

    LogisticRegressor fits a linear model for p predictor variables 
    with coefficients b = (b_1,..., b_p) for each class to the 
    log-odds (aka logit) by minimizing the associated log-likelyhood.

    Parameters:
        fit_intercept: (bool, default = True)
            If False, it sets the bias term b_0 = 0, i.e. the intercept, 
            for all classes and no intercept is used in the optimisation.

        tolerance: (float, default = 1e-4)
            The precision of the optimisation method.

        max_iter: (float)
            The precision of the optimisation method.

    Attributes:
        fit_intercept_ (bool): 
            Stores the fit_intercept parameter.

        tolerance_ (float):
            The tolerance criteria used to stop the 
            optimisation method.

        max_iter_ (int):
            The maximum number of iterations used to 
            stop the optimisation method.

        intercept_ (float): 
            The intercept coefficient.

        coef_ (ndarray): array of shape (p_predictors,).
            The predictor's coefficient estimators.

        classes_ (ndarray): array of shape (n_classes,) 
            The list of class labels as seeing in fit.

        n_classes_ (int):  
            The number of class labels as seeing in fit.

        n_iter_ (int):
            The number of iterations taken by the 
            optimisation method to converge within the 
            chosen tolerance.

    Methods:
        fit(X, y):
            Fits the linear model.

        predict_proba(X):
            Probabilities estimates from the samples in X.

        predict(X):
            Predicts the classes from the samples in X.

        scores(X, y):
            Returns a number of scores for multiclass classification, 
            such as prevalence, accuracy, precision and true/false 
            positive rates.

        confusion_matrix(X, y):
            Returns the confusion matrix with a summary of the
            performance of the classification model.            
    
        get_params():
            Get the parameters (b_0,...,b_p) of the estimator
            for each class.

        regression_report(X, y):
            Returns a report on the statistical analysis 
            of the estimators's coefficients, such as the 
            standard error, z-statistics and p-value.
    """

    def __init__(
            self, 
            fit_intercept: bool = True, 
            tolerance: float = 1e-4, 
            max_iter: int = 100
            ) -> None:
        """
        Initialises the LogisticRegressor with parameters.

        Keyword arguments:
            fit_intercept (bool, default = True):
                If False, it sets the intercept to zero and it 
                is not used in the minimization.

            tolerance (float, default = 1e-4):
                The tolerance for the stop criteria used in the 
                opmisation method.

            max_iter (int, default = 100):
                The max number of iterations used in the 
                optimisation method.
        """

        self.fit_intercept = fit_intercept
        self.tolerance = tolerance
        self.max_iter = max_iter        
        
        self.intercept_ = None
        self.coef_ = None
        self.classes_ = None
        self.n_classes_ = 0
        self.n_iter_ = 0 


    def fit(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
            sample_weight: np.ndarray = None
            ) -> None:
        """
        Fits the logistic model.

        
        Keyword arguments:
            X (np.ndarray): array of shape (n_samples, n_predictors).
                The predictor's data.

            y (np.ndarray): array of shape (n_samples,).
                The dependent variable's data.

            sample_weight (np.ndarray, default = None): array of shape (n_samples,) 
                The weight of each sample.

        Returns:
            self:
                The fitted estimator.
        """

        check_same_number_of_rows(X, y)
        n_samples, n_predictors = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size

        y_one_hot_encoded = one_hot_encoder(y, self.classes_)

        if sample_weight:
            w = sample_weight
        else:
            w = np.ones(y.shape)
        
        if self.fit_intercept:
            X_ = np.concatenate((np.ones((n_samples, 1)), X), axis = 1)
            b0_hat = np.zeros((n_predictors + 1, self.n_classes_))
        else:
            X_ = X.copy()
            b0_hat = np.zeros((n_predictors, self.n_classes_))
                      

        opt = NewtonOptimiser(X_, y_one_hot_encoded, w)
        b_hat = opt.optimise(b0_hat, self.tolerance, self.max_iter)
        self.n_iter_ = opt.n_iter_

        if self.fit_intercept:
            self.intercept_ = b_hat[0, :]
            self.coef_ = b_hat[1:, :]
        else:
            self.intercept_ = np.zeros(self.n_classes_)
            self.coef_ = b_hat[:, :]


    def predict_proba(
            self, 
            X: np.ndarray
            ) -> np.ndarray: 
        """
        Probabilities estimates for the reference class 1 
        from the samples in X.

        
        Keyword arguments:
            X (np.ndarray): array of shape (n_samples, n_predictors).
                The predictor's data.

        Returns:
            np.ndarray: array of shape (n_samples,).
        """
        
        n_samples = X.shape[0]
        X_ = np.concatenate((np.ones((n_samples, 1)),X), axis=1)
            
        b_hat = np.append(self.intercept_.reshape(1, -1), self.coef_, axis = 0)

        proba_hat = softmax(X_ @ b_hat)

        return proba_hat


    def predict(
            self, 
            X: np.ndarray
            ) -> np.ndarray: 
        """
        Predicts the values of the dependent variable using 
        the fitted linear model.

        Keyword arguments:
            X (np.ndarray): array of shape (n_samples, n_predictors).
                The predictor's data.

        Returns:
            np.ndarray: array of shape (n_samples,).
        """
        
        n_samples = X.shape[0]
        X_ = np.concatenate((np.ones((n_samples, 1)),X), axis=1)
            
        b_hat = np.append(self.intercept_.reshape(1, -1), self.coef_, axis = 0)
        proba_hat = softmax(X_ @ b_hat)
        y_pred = self.classes_[np.argmax(proba_hat, axis = 1)]
        
        return y_pred

    def scores(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
            sample_weight: np.ndarray = None
            ) -> None:
        """
        Prints the prevalence, accuracy, precision and true/false 
        negative/positive rates.

        Keyword arguments:
            X (np.ndarray): array of shape (n_samples, n_predictors).
                The predictor's data.

            y (np.ndarray): array of shape (n_samples,).
                The dependent variable's data.

            sample_weight (np.ndarray, default = None): array of shape (n_samples,) 
                The weight of each sample.
        """

        check_same_number_of_rows(X, y)

        n_samples = y.shape[0]
        b_hat = np.append(self.intercept_.reshape(1, -1), self.coef_, axis = 0)

        if sample_weight:
            w = sample_weight
        else:
            w = np.ones(y.shape)
        
        X_ = np.concatenate((np.ones((n_samples, 1)),X), axis=1)
        proba_hat = softmax(X_ @ b_hat)
        y_pred = self.classes_[np.argmax(proba_hat, axis = 1)]

        y_ohe = one_hot_encoder(y, self.classes_)
        y_pred_ohe = one_hot_encoder(y_pred, self.classes_)

        t = (y_pred_ohe == y_ohe).prod(axis = 1)
        p = (np.diag(w) @ y_ohe).sum(axis = 0)
        tp = (np.diag(w*t) @ y_pred_ohe).sum(axis = 0)
        fp = (np.diag(w*(1 - t)) @ y_pred_ohe).sum(axis = 0)
        fn = (np.diag(w*(1 - t)) @ y_ohe).sum(axis = 0)

        score_table = PrettyTable(["quantity"] + ["class " + str(label) for label in self.classes_] + ["all classes"])
        score_table.add_row(["prevalence"] + [value for value in np.around(p/n_samples, 4)] + [" --- "])
        score_table.add_row(["precision"] + [value for value in np.around(tp/(tp + fp), 4)] + [" --- "])
        score_table.add_row(["recall"] + [value for value in np.around(tp/(tp + fn), 4)] + [" --- "])
        score_table.add_row(["true positive rate"] + [value for value in np.around(tp/p, 4)] + [" --- "])
        score_table.add_row(["false positive rate"] + [value for value in np.around(fp/p, 4)] + [" --- "])
        score_table.add_row(["accuracy"] + self.n_classes_*[" --- "] + [np.around(tp.sum()/n_samples, 4)])

        print(score_table)

        return

    def confusion_matrix(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
            sample_weight: np.ndarray = None
            ) -> None:
        """
        Prints the confusion matrix comparing the predicted and the 
        actual classes for the binary classification problem.

        Keyword arguments:
            X (np.ndarray): array of shape (n_samples, n_predictors).
                The predictor's data.

            y (np.ndarray): array of shape (n_samples,).
                The dependent variable's data. 

            sample_weight (np.ndarray, default = None): array of shape (n_samples,) 
                The weight of each sample.
        """

        check_same_number_of_rows(X, y)

        n_samples = y.shape[0]
        b_hat = np.append(self.intercept_.reshape(1, -1), self.coef_, axis = 0)

        if sample_weight:
            w = sample_weight
        else:
            w = np.ones(y.shape)
        
        X_ = np.concatenate((np.ones((n_samples, 1)),X), axis=1)
        proba_hat = softmax(X_ @ b_hat)
        y_pred = self.classes_[np.argmax(proba_hat, axis = 1)]
        print(y_pred.size)

        confusion_matrix = PrettyTable(["***"] + ["predicted " + str(label) for label in self.classes_])

        for label1 in self.classes_:
            row = ["true " + str(label1)]
            mask = np.where(y == label1)
            y_pred_masked = y_pred[mask]
            w_masked = w[mask]
            for label2 in self.classes_:
                count = np.sum(w_masked*(y_pred_masked == label2).astype(int))
                row.append(count)

            confusion_matrix.add_row(row)
            
        print(confusion_matrix)
            
        return

    def get_params(self) -> dict:
        """
        Get the logistic regression coefficients for the estimator.

        Returns:
            C (dict): a dictionary with the coefficient names as keys
            and their correspondent values. 
        """

        params_dict = {'intercept': self.intercept_}
        for id_, c in enumerate(self.coef_):
            params_dict.update({'coef_' + str(id_ + 1): c})

        return params_dict

    def regression_report(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
            sample_weight: np.ndarray = None
            ) -> None:
        """
            Prints a report on the statistical analysis 
            for each of the estimators's coefficients. 
            It gives the standard error, the z-statistics and 
            p-value to reject the hypothesis that the coefficient 
            is null assuming all the other coefficients are fixed.

        Keyword arguments:
            X (np.ndarray): array of shape (n_samples, n_predictors).
                The predictor's data.

            y (np.ndarray): array of shape (n_samples,).
                The dependent variable's data.

            sample_weight (np.ndarray, default = None): array of shape (n_samples,) 
                The weight of each sample.
        """

        check_same_number_of_rows(X, y)

        n_samples = y.shape[0]
        
        if sample_weight:
            w = sample_weight
        else:
            w = np.ones(y.shape)

        if self.fit_intercept:
            X_ = np.concatenate((np.ones((n_samples, 1)),X), axis=1)
            b_hat = np.append(self.intercept_.reshape(1, -1), self.coef_, axis = 0)
        else:
            X_ = X
            b_hat = self.coef_

        H_inv = np.linalg.inv(hessian_func(X_, w, b_hat))

        std_errors = np.sqrt(np.diag(H_inv))
        z_stats = b_hat[:, : self.n_classes_ - 1].flatten('F')/std_errors
        z_stats = np.reshape(z_stats, (b_hat.shape[0], b_hat.shape[1] - 1), 'F')
        p_values = 1 - erf(np.abs(z_stats)/np.sqrt(2.))

        report_table = PrettyTable(["*****", "coefficient", "std. error", "z-statistics", "p-value"])
        if self.fit_intercept:
            report_table.add_row(
                ["intercept", 
                 np.round(self.intercept_[: self.n_classes_ - 1], 4), 
                 np.round(std_errors[0], 4),
                 np.round(z_stats[0, : self.n_classes_ - 1], 4),
                 np.round(p_values[0, : self.n_classes_ - 1], 4)
                ]
            )

            for i, (coef, std_err, z, p_value) in enumerate(
                zip(
                    self.coef_[:, :self.n_classes_ - 1], 
                    std_errors[1:], 
                    z_stats[1:, :self.n_classes_ - 1], 
                    p_values[:, :self.n_classes_ - 1]
                    )
                ):
                
                report_table.add_row(
                    [f"coef_{i + 1}", 
                    np.round(coef, 4), 
                    np.round(std_err, 4),
                    np.round(z, 4),
                    np.round(p_value, 4)
                    ]
                )

        else:
            report_table.add_row([f"intercept", np.zeros(self.n_classes_ - 1), "----", "----", "----"])

            for i, (coef, std_err, z, p_value) in enumerate(
                zip(
                    self.coef_[:, :self.n_classes_ - 1], 
                    std_errors, z_stats[:, :self.n_classes_ - 1], 
                    p_values[:, :self.n_classes_ - 1]
                    )
                ):

                report_table.add_row(
                    [f"coef_{i + 1}", 
                    np.round(coef, 4), 
                    np.round(std_err, 4),
                    np.round(z, 4),
                    np.round(p_value, 4)
                    ]
                )
            
        print(report_table)

        return
