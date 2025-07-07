from functools import partial
import numpy as np
from ._utilities import check_same_number_of_rows
from ._utilities import softmax, one_hot_encoder
from ._optimiser import NewtonOptimiser, hessian_func
from scipy.stats import chi2
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

        intercept_ (ndarray): array of shape (n_classes,) 
            The intercept coefficient estimators.

        coef_ (ndarray): array of shape (n_predictors, n_classes).
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

        Returns (self):
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

        Returns (ndarray):
            Array of shape (n_samples,).
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

        Returns (ndarray):
            Array of shape (n_samples,).
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
            ) -> PrettyTable:
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

        Returns (PrettyTable):
            The score table.
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


        return score_table

    def confusion_matrix(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
            sample_weight: np.ndarray = None
            ) -> PrettyTable:
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

        Returns (PrettyTable):
            The matrix confusion table.
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
                        
        return confusion_matrix

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
            ) -> PrettyTable:
        """
        Prints a report on the statistical analysis 
        for each of the estimators's coefficients. 
        It gives the standard error, the z-statistics and 
        p-value to reject the hypothesis that the coefficients
        corresponding to a given predictor is null for all 
        classes, assuming all the other coefficients 
        are fixed.

        Keyword arguments:
            X (np.ndarray): array of shape (n_samples, n_predictors).
                The predictor's data.

            y (np.ndarray): array of shape (n_samples,).
                The dependent variable's data.

            sample_weight (np.ndarray, default = None): array of shape (n_samples,) 
                The weight of each sample.

        Returns (PrettyTable):
            The table with the regression report.
        """

        check_same_number_of_rows(X, y)

        n_samples, n_predictors = X.shape
        
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

        std_errors = np.sqrt(np.diag(H_inv)/n_samples)
        z_stats = b_hat[:, : self.n_classes_ - 1].flatten('F')/std_errors

        std_errors = np.append(std_errors, np.zeros(n_predictors + self.fit_intercept))
        z_stats = np.append(z_stats, np.zeros(n_predictors + self.fit_intercept))

        w_stats = np.zeros(n_predictors + self.fit_intercept)
        for i in range(n_predictors + self.fit_intercept):
            beta_hat = b_hat[i, : self.n_classes_ - 1].reshape(-1,1)
            Sigma_i = np.zeros((self.n_classes_ - 1, self.n_classes_ - 1))
            for k in range(self.n_classes_ - 1):
                for l in range(self.n_classes_ - 1):
                    Sigma_i[k,l] = H_inv[k*(n_predictors + self.fit_intercept) + i, l*(n_predictors + self.fit_intercept) + i]/n_samples
         
            w_stats[i] = beta_hat.T @ np.linalg.inv(Sigma_i) @ beta_hat

        p_values = 1 - chi2._cdf(w_stats, self.n_classes_ - 1)

        report_table = PrettyTable(["*****", "coefficient", "std. error", "z-statistics", "p-value"])
        if self.fit_intercept:
            idx_ = np.array([k*(n_predictors + 1) for k in range(self.n_classes_)])
            report_table.add_row(
                ["intercept", 
                 np.round(self.intercept_[: self.n_classes_], 4), 
                 np.round(std_errors[idx_], 4),
                 np.round(z_stats[idx_], 4),
                 np.round(p_values[0], 4)
                ]
            )

            for i in range(n_predictors):
                idx_ = np.array([k*(n_predictors + 1) + i + 1 for k in range(self.n_classes_)])
                report_table.add_row(
                    [f"coef_{i + 1}", 
                    np.round(self.coef_[i, : self.n_classes_], 4), 
                    np.round(std_errors[idx_], 4),
                    np.round(z_stats[idx_], 4),
                    np.round(p_values[i], 4)
                    ]
                )


        else:
            report_table.add_row([f"intercept", np.zeros(self.n_classes_ - 1), "----", "----", "----"])

            for i in range(n_predictors):
                idx_ = np.array([k*n_predictors + i for k in range(self.n_classes_)])
                report_table.add_row(
                    [f"coef_{i + 1}", 
                    np.round(self.coef_[i, : self.n_classes_], 4), 
                    np.round(std_errors[idx_], 4),
                    np.round(z_stats[idx_], 4),
                    np.round(p_values[i], 4)
                    ]
                )
            

        return report_table
